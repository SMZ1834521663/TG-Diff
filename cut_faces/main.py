import os
import gc
import sys
import time
import argparse
import signal
import traceback
import threading
import subprocess
import platform
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,TimeoutError,as_completed

from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_EDGE,TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Face,TopoDS_Wire,TopoDS_Shell,TopoDS_Solid,topods_Edge,topods_Vertex
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer

if platform.system() != "Linux":
    from OCC.Display.SimpleGui import init_display
use_display = True if platform.system() != "Linux" else False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cut_faces.utils_occ import get_vertice_from_edge,are_vertice_same,are_edges_same,is_edge_in,get_edges_from_face,write_step_file,write_stl_file,sewing_cutted_faces,fix_face
from cut_faces.utils_process import check_brep_topo_validity, check_brep_validity_by_edge, del_surf_from_edge_degree_3, del_surf_from_vert_degree_1, extend_faces_and_make_params, fit_bspline_surface, get_adj_faces_distance,dprint,final_fuse
from cut_faces.utils_split import split_edge_from_points,split_face_from_edges
from cut_faces.utils_align import align_solid_surfs,extract_uv_aligned_edges
from cut_faces.config import cfg


if (cfg.mode==1 or cfg.mode == 2) and use_display:
    display, start_display, add_menu, add_function_to_menu = init_display()
else:
    display = None
    start_display = None


# for cut_face_and_select
def del_hang_edge(edges,vertice,main_index):
    vertice_to_edges = defaultdict(list)
    for v in vertice:
        vertice_to_edges[v] = []

    for e in edges:
        v1,v2 = get_vertice_from_edge(e)
        get_v1,get_v2 = False,False
        for v in vertice:
            if are_vertice_same(v,v1,tolerance=1e-3):
                vertice_to_edges[v].append(e)
                get_v1 = True
            elif are_vertice_same(v,v2,tolerance=1e-3):
                vertice_to_edges[v].append(e)
                get_v2 = True
            if get_v1 and get_v2:
                break
        else:
            dprint([main_index,"no v1v2"],cfg.print)
    
    for v,es in vertice_to_edges.items():
        if len(es)==1:
            if es[0] in edges:
                edges.remove(es[0])
            if v in vertice:
                vertice.remove(v)
    return edges,vertice


def cut_face_and_select(main_face_params,cut_faces_params,use_ef_section,aligned_edges=None,skip_cut=False):
    if aligned_edges is None:
        aligned_edges = []
    main_index = list(main_face_params.keys())[0]
    dprint(["cutting:",main_index],cfg.print)
    
    #################### edge section ####################
    face_not_section_cut_index = []
    main_face_orign = list(main_face_params.values())[0][0]
    main_face_extended = list(main_face_params.values())[0][1]

    ##### skip-cut-for-fully-aligned-face start #####
    if bool(skip_cut):
        return [main_face_orign], False, main_index, face_not_section_cut_index

    section_edges = []
    ##### preset-section-edges start #####
    for e in aligned_edges:
        section_edges.append([topods_Edge(e), main_face_extended, main_index, True])
        
    ##### preset-section-edges end #####
    for one_cut_face_params in cut_faces_params:
        cut_index = list(one_cut_face_params.keys())[0]
        cut_face_orign,cut_face_extended = list(one_cut_face_params.values())[0]
        section = BRepAlgoAPI_Section(main_face_extended, cut_face_extended, False)
        section.SetFuzzyValue(1e-5)  
        section.Build()
        section_edge_compound = section.Shape()
        explorer = TopExp_Explorer(section_edge_compound, TopAbs_EDGE)
        section_count = 0
        temp_section_edges=[]
        while explorer.More():
            temp_section_edges.append(topods_Edge(explorer.Current()))
            explorer.Next()
            section_count+=1
        for e in temp_section_edges:
            section_edges.append([e,cut_face_extended,cut_index,False])
        if section_count==0:
            face_not_section_cut_index.append(cut_index)  

    #################### vertex section ####################
    # Kernel-limited near-intersection failure
    if not use_ef_section: # ee section
        tol = 1e-4
        for now_num in range(4):
            section_points=[]
            for i in range(len(section_edges)):
                for j in range(i+1,len(section_edges)):
                    section = BRepAlgoAPI_Section(section_edges[i][0], section_edges[j][0], False)
                    section.SetFuzzyValue(tol)  
                    section.Build()
                    section_point_compound = section.Shape()
                    explorer = TopExp_Explorer(section_point_compound, TopAbs_VERTEX)
                    while explorer.More(): 
                        section_points.append(topods_Vertex(explorer.Current()))
                        explorer.Next()

            if len(section_points)<len(section_edges):
                tol=tol*10
            else:
                break
 
    if use_ef_section or len(section_points)<len(section_edges): # ef section
        dprint(["high per edge"],cfg.print)
        section_points=[]
        for i in range(len(section_edges)):
            for j in range(i+1,len(section_edges)):
                section = BRepAlgoAPI_Section(section_edges[i][0], section_edges[j][1], False)
                section.SetFuzzyValue(1e-4)  
                section.Build()
                section_point_compound = section.Shape()
                explorer = TopExp_Explorer(section_point_compound, TopAbs_VERTEX)
                while explorer.More(): 
                    section_points.append(topods_Vertex(explorer.Current()))
                    explorer.Next()
    section_points_duplicated=[]
    for v in section_points:
        if len(section_points_duplicated)==0 :
            section_points_duplicated.append(v)
            continue
        for v_in in section_points_duplicated:
            if are_vertice_same(v,v_in,tolerance=1e-4): 
                break
        else:
            section_points_duplicated.append(v)


    #################### split edge ####################
    cutted_edges_all = []
    for e in section_edges:
        e =e[0]
        splitted_edges = split_edge_from_points(e,section_points_duplicated,tolerance=1e-3)
        cutted_edges_all.extend(splitted_edges)
    cutted_edges_all,section_points_duplicated = del_hang_edge(cutted_edges_all,section_points_duplicated,main_index)
    
    selected_edges=[]
    for cutted_edge in cutted_edges_all:
        c, first, last = BRep_Tool.Curve(cutted_edge)
        pnt_first,pnt_last = gp_Pnt(),gp_Pnt()
        c.D0(first,pnt_first)
        c.D0(last,pnt_last)
        get_first,get_last=False,False
        for v in section_points_duplicated:
            pnt_v=BRep_Tool.Pnt(v)
            if get_first == False and pnt_first.Distance(pnt_v)<1e-3 :
                get_first=True
            if get_last == False and pnt_last.Distance(pnt_v)<1e-3 :
                get_last=True
        if get_first and get_last:
            selected_edges.append(cutted_edge)

    selected_edges_duplicated=[e for e in selected_edges]
    same_edges=[]
    for e in selected_edges:
        for e_in in selected_edges:
            if e != e_in:
                if are_edges_same(e,e_in,tolerance=1e-4) and e in selected_edges_duplicated:
                    selected_edges_duplicated.remove(e)
                    for e_same in same_edges:
                        if are_edges_same(e,e_same,tolerance=1e-4):
                            break
                    else:same_edges.append(e)
                if is_edge_in(e,e_in,tolerance=1e-4) and e in selected_edges_duplicated: 
                    selected_edges_duplicated.remove(e)

    selected_edges_duplicated.extend(same_edges)

    #################### split face ####################
    cutted_faces=split_face_from_edges(main_face_extended,selected_edges_duplicated)

    count_array = np.zeros(len(selected_edges))
    selected_faces=[] # [topods_face,count_array_one_face]
    for current_face in cutted_faces:  
        current_face = fix_face(current_face)
        count_array_one_face = np.zeros(len(selected_edges))
        not_flag = False
        current_face_edges = get_edges_from_face(current_face)
        for edge in current_face_edges:
            for k in range(len(selected_edges)):
                if is_edge_in(selected_edges[k],edge,tolerance=1e-3): 
                    count_array_one_face[k]+=1
                    break
            else:
                not_flag=True
                break
        if not_flag:
            continue
        count_array+=count_array_one_face
        selected_faces.append([current_face,count_array_one_face])

    sorted_selected_faces = sorted(selected_faces, key=lambda x: sum(x[1]),reverse=True)
    del_selected_faces=[]
    for i in range(len(sorted_selected_faces)):
        mask_i = sorted_selected_faces[i][1].astype(bool) 
        if np.all((count_array[mask_i]-sorted_selected_faces[i][1][mask_i])>=1): 
            for j in range(len(sorted_selected_faces)):
                if i==j:continue
                mask_j = sorted_selected_faces[j][1].astype(bool)
                if np.all((count_array[mask_j]-sorted_selected_faces[j][1][mask_j])>=1):
                    if np.any(sorted_selected_faces[i][1].astype(bool) & sorted_selected_faces[j][1].astype(bool)):
                        break
            else:
                del_selected_faces.append(sorted_selected_faces[i])
                count_array-=sorted_selected_faces[i][1]

    for del_selected_face in del_selected_faces:
        sorted_selected_faces.remove(del_selected_face)

    return_faces=[f for f,arr in sorted_selected_faces]
    
    error_flag = False
    if len(return_faces)==0:
        dprint([main_index,"error"],cfg.print)
        error_flag = True
    
    return return_faces,error_flag,main_index,face_not_section_cut_index


########################################################
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")

def safe_kill(pid):
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"Kill failed ({pid}): {e}")

def initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_one_face(arguments):
    now_face, cut_faces, use_ef_section, preset_section_edges, skip_cut = arguments
    stop_event = threading.Event()
    def killer():
        timeout = 15+len(cut_faces)*4
        for _ in range(int(timeout*10)):
            if stop_event.is_set():
                return 
            time.sleep(0.1)
        print("kill")
        safe_kill(os.getpid())
    
    t=threading.Thread(target=killer, daemon=True)
    t.start()
 
    result=cut_face_and_select(
        now_face,
        cut_faces,
        use_ef_section,
        aligned_edges=preset_section_edges,
        skip_cut=skip_cut,
    )
    stop_event.set() 

    return result

def process_one_solid(
        surf,
        surf_mask,
        adj_matrix,
        index,
        save_num,
        save_dir,
        data_path,
        face_workers,
        use_fit_plane,
        use_fit_cylinder,
        use_fit_cone,
        use_fit_torus,
        use_fit_sphere,
        use_align,
        use_final_fuse,
    ):

    aligned_decay_scale = 0.12
    aligned_extend_scale = 0.8
    max_decay = 0.18
    cut_params=[
        [0.6,0.09,5e-2,0], 
        [0.6,0.12,5e-2,0],
        [0.6,0.16,5e-2,0],
        [0.6,0.10,5e-2,2],
        # [1.5,0.20,5e-2,0],
        ]
    sew_params=[
        [1e-3],
    ]

    mode = cfg.mode  #1.only view, 2.single process, 3.multi process
    global display, start_display
    # display/start_display are shared globals
    # keep one naming scheme
    if cfg.mode == 3 and use_display:
        display, start_display, add_menu, add_function_to_menu = init_display()

    use_cylinder_decay = True
    cylinder_decay_once = 0.10
    use_cone_decay = True
    cone_decay_once = 0.10
    use_torus_decay = True
    torus_decay_once = [0.10,0.10]
    use_sphere_decay = True
    sphere_decay_once = [0.10,0.10]

    #################################################### start
    # print("now index",index)
    os.makedirs(data_path + "/" +"complete",exist_ok=True)
    with open(data_path + "/" + "complete" + "/" + "{}.txt".format(index), "a+", encoding="utf-8") as f:
        pass

    # align
    aligned_edge_flags = [[0, 0, 0, 0] for _ in range(len(surf))]
    alignment_groups = []
    if use_align:
        try:
            surf, alignment_groups, aligned_edge_flags = align_solid_surfs(surf, surf_mask, adj_matrix)
            if alignment_groups:
                dprint([f"solid {index}: aligned face groups={alignment_groups}"],cfg.print)
                for group in alignment_groups:
                    for face_idx in group:
                        dprint([f"solid {index}: face {face_idx} aligned edges [u-,u+,v-,v+]={aligned_edge_flags[face_idx]}"],cfg.print)
        except Exception as e:
            dprint([f"solid {index}: align skipped ({type(e).__name__}: {e})"],cfg.print)
    aligned_face_mask = np.zeros((len(surf),), dtype=np.bool_)
    for group in alignment_groups:
        for face_idx in group:
            if 0 <= face_idx < len(aligned_face_mask):
                aligned_face_mask[face_idx] = True

    all_results = []
    fit_tolerance_array=np.ones((len(surf),)) *cut_params[0][2] 
    use_ef_section_array=np.zeros((len(surf),),dtype=np.bool_) 
    cylinder_decay_array = np.zeros((len(surf),),dtype=np.float32)
    cone_decay_array = np.zeros((len(surf),),dtype=np.float32)
    torus_decay_array = np.zeros((len(surf),2),dtype=np.float32)
    sphere_decay_array = np.zeros((len(surf),2),dtype=np.float32)
    for now_num in range(len(cut_params)):
        try :
            extend_length_dis = cut_params[now_num][0]
            extend_length_const = cut_params[now_num][1]
            spline_fit_tolerance = cut_params[now_num][2]
            reset_flag = cut_params[now_num][3]
            if reset_flag==1: 
                cylinder_decay_array = np.zeros((len(surf),),dtype=np.float32)
                cone_decay_array = np.zeros((len(surf),),dtype=np.float32)
                torus_decay_array = np.zeros((len(surf),2),dtype=np.float32)
                sphere_decay_array = np.zeros((len(surf),2),dtype=np.float32)
            elif reset_flag==2:
                cylinder_decay_array[:] = max_decay
                cone_decay_array[:] = max_decay
                torus_decay_array[:,0],torus_decay_array[:,1] = max_decay,max_decay
                sphere_decay_array[:,0],sphere_decay_array[:,1] = max_decay,max_decay

            fit_cylinder_decay_array = cylinder_decay_array.copy()
            fit_cone_decay_array = cone_decay_array.copy()
            fit_torus_decay_array = torus_decay_array.copy()
            fit_sphere_decay_array = sphere_decay_array.copy()
            if np.any(aligned_face_mask):
                fit_cylinder_decay_array[aligned_face_mask] *= aligned_decay_scale
                fit_cone_decay_array[aligned_face_mask] *= aligned_decay_scale
                fit_torus_decay_array[aligned_face_mask] *= aligned_decay_scale
                fit_sphere_decay_array[aligned_face_mask] *= aligned_decay_scale

            recon_faces = fit_bspline_surface(
                surf,
                surf_mask,
                fit_tolerance_array,
                use_fit_plane,
                use_fit_cylinder,
                use_fit_cone,
                use_fit_torus,
                use_fit_sphere,
                use_cylinder_decay,
                use_cone_decay,
                use_torus_decay,
                use_sphere_decay,
                fit_cylinder_decay_array,
                fit_cone_decay_array,
                fit_torus_decay_array,
                fit_sphere_decay_array,
            )

            ##### aligned-edge extension guard end #####
            torus_sphere_edge_flags = [
                aligned_edge_flags[i] if recon_faces[i]["type"] in {"torus", "sphere"} else [0, 0, 0, 0]
                for i in range(len(recon_faces))
            ]
            max_distance = get_adj_faces_distance(recon_faces,adj_matrix)
            extend_scales = np.ones((len(recon_faces),), dtype=np.float32)
            if np.any(aligned_face_mask):
                extend_scales[aligned_face_mask[:len(recon_faces)]] = aligned_extend_scale
            all_face_params = extend_faces_and_make_params(
                recon_faces,
                max_distance,
                extend_length_dis,
                extend_length_const,
                edge_flags=torus_sphere_edge_flags,
                extend_scales=extend_scales,
            )

            # adj face
            face_and_cut_faces = []
            aligned_adj = np.zeros((len(recon_faces), len(recon_faces)), dtype=np.bool_)
            for group in alignment_groups:
                for i in group:
                    for j in group:
                        if i != j:
                            aligned_adj[i, j] = True
            for i in range(len(recon_faces)):
                connect_faces_params_list = []
                for j in range(len(recon_faces)):
                    aligned_torus_sphere_pair = aligned_adj[i, j] and any(torus_sphere_edge_flags[i]) and any(torus_sphere_edge_flags[j])
                    if i!=j and adj_matrix[i][j]==1 and not aligned_torus_sphere_pair:
                        connect_faces_params_list.append({j:all_face_params[j]})
                main_face_params = {i:all_face_params[i]}
                use_ef_section = use_ef_section_array[i]
                preset_section_edges = extract_uv_aligned_edges(all_face_params[i][0], torus_sphere_edge_flags[i])
                skip_cut = len(preset_section_edges) >= 4 and all(bool(x) for x in torus_sphere_edge_flags[i])
                face_and_cut_faces.append([main_face_params, connect_faces_params_list, use_ef_section, preset_section_edges, skip_cut])

            ###################################################################
            is_error = False 
            if mode==1:
                shape_hash = {}
                for i in range(len(recon_faces)):
                    shape_hash[hash(all_face_params[i][1])] = "face{}".format(i)
                    display.DisplayShape(all_face_params[i][1], update=True,color=0,transparency=0.5)
                def click_callback(shape, *kwargs):
                    for shape in shape:
                        if isinstance(shape,TopoDS_Face):
                            print(shape_hash[hash(shape)])
                display.register_select_callback(click_callback)
            elif mode==2: 
                results = []
                display.EraseAll()
                for i in range(len(recon_faces)):
                    result = process_one_face(face_and_cut_faces[i])
                    if result!=None:
                        return_faces,error_flag,main_index,face_section_error_cut_index = result
                        results.append(result)
                        for f in return_faces:
                            display.DisplayShape(f, update=True,color=0,transparency=0.5)
            elif mode==3: 
                try: 
                    with ProcessPoolExecutor(max_workers=face_workers) as executor:
                        
                        futures = {executor.submit(process_one_face, face_and_cut_faces[i]): i for i in range(len(face_and_cut_faces))}
                        results = []
                        for future in tqdm(as_completed(futures), total=len(face_and_cut_faces),disable=True):
                            try:
                                result = future.result(timeout=60) 
                                if result is not None:
                                    results.append(result)
                            except TimeoutError:
                                print(f"over time")
                                future.cancel() 
                                executor.shutdown(wait=False,cancel_futures=True)
                                is_error = True
                                break
                            except Exception as e:
                                future.cancel() 
                                is_error = True
                                print(f"shutdown")
                                executor.shutdown(wait=False,cancel_futures=True)
                                traceback.print_exc()
                    gc.collect()     
                except:
                    executor.shutdown(wait=False,cancel_futures=True)
                    is_error = True
                    gc.collect()

            if (mode==2 or mode==3) and len(results)>0: 
                success = False
                error_sum = 0 

                all_main_index = [i for i in range(len(surf))]
                decay_change_over = []
                finetune_change_over = []
                for res_faces, error_flag, main_index, face_section_error_cut_index in results:  
                    all_main_index.remove(main_index)
                    error_sum += int(error_flag)
                    if error_flag==True :
                        # bspline low persion  (useless)
                        if now_num<len(cut_params)-1: 
                            fit_tolerance_array[main_index] = cut_params[now_num+1][2] 
                            for idx in face_section_error_cut_index:
                                fit_tolerance_array[idx] = cut_params[now_num+1][2]

                        # use ef section                               
                        use_ef_section_array[main_index] = True 

                        # decay
                        if main_index not in decay_change_over:
                            cylinder_decay_array[main_index]= np.clip(cylinder_decay_array[main_index] + cylinder_decay_once,None,max_decay)
                            cone_decay_array[main_index]= np.clip(cone_decay_array[main_index] + cone_decay_once,None,max_decay)
                            torus_decay_array[main_index]= np.clip(torus_decay_array[main_index] + torus_decay_once,None,[max_decay,max_decay])
                            sphere_decay_array[main_index]= np.clip(sphere_decay_array[main_index] + sphere_decay_once,None,[max_decay,max_decay])

                        connect_faces_params_list = face_and_cut_faces[main_index][1]
                        decay_change_over.append(main_index)
                        for cut_face_params in connect_faces_params_list:
                            j = list(cut_face_params.keys())[0]
                            if j not in decay_change_over:
                                cylinder_decay_array[j]= np.clip(cylinder_decay_array[j] + cylinder_decay_once,None,max_decay)
                                cone_decay_array[j]= np.clip(cone_decay_array[j] + cone_decay_once,None,max_decay)
                                torus_decay_array[j]= np.clip(torus_decay_array[j] + torus_decay_once,None,[max_decay,max_decay])
                                sphere_decay_array[j]= np.clip(sphere_decay_array[j] + sphere_decay_once,None,[max_decay,max_decay])
                                decay_change_over.append(j)
                
                # over time patten
                for other_main_index in all_main_index:
                    if now_num<len(cut_params)-1:
                        fit_tolerance_array[other_main_index] = cut_params[now_num+1][2]
                    if other_main_index not in decay_change_over:
                        cylinder_decay_array[other_main_index]= np.clip(cylinder_decay_array[other_main_index] + cylinder_decay_once,None,max_decay)
                        cone_decay_array[other_main_index]= np.clip(cone_decay_array[other_main_index] + cone_decay_once,None,max_decay)
                        torus_decay_array[other_main_index]= np.clip(torus_decay_array[other_main_index] + torus_decay_once,None,[max_decay,max_decay])
                        sphere_decay_array[other_main_index]= np.clip(sphere_decay_array[other_main_index] + sphere_decay_once,None,[max_decay,max_decay])
                        decay_change_over.append(other_main_index)
                    
                #final process
                if error_sum==0 and not is_error:
                    results = sorted(results,key=lambda x:x[2],reverse=False) 
                    results = del_surf_from_edge_degree_3(results)
                    results = del_surf_from_vert_degree_1(results,adj_matrix)

                    topo_valid = check_brep_topo_validity(results)
                    if topo_valid !=True:
                        dprint(["topo error"],cfg.print)

                    edge_valid = check_brep_validity_by_edge(results)
                    if edge_valid != True:
                        dprint(["edge degree 1 error"],cfg.print)
                                    
                    # for display and save
                    all_res_faces = []  
                    for result in results:
                        all_res_faces.extend(result[0])

                    if edge_valid == True and topo_valid:
                        if use_final_fuse:
                            all_res_faces = []
                            for result in results:
                                new_face = final_fuse(result[0],all_face_params[main_index][1])
                                all_res_faces.append(new_face)
                        
                        # sewing
                        for now_num in range(len(sew_params)):
                            try:
                                sewing_tolerance = sew_params[now_num][0]
                                solid = sewing_cutted_faces(all_res_faces,sewing_tolerance)
                                break
                            except:
                                dprint(["sew error"],cfg.print)
                        #写入文件
                        if isinstance(solid,TopoDS_Shell) or isinstance(solid,TopoDS_Solid):
                            write_step_file(solid, '{}/{}_{}_{}.step'.format(save_dir,len(results),save_num,index))
                            write_stl_file(solid, '{}/{}_{}_{}.stl'.format(save_dir,len(results),save_num,index), linear_deflection=0.001, angular_deflection=0.5)
                            success = True
                        else:
                            dprint(["unclosed," + "retry:{}".format(now_num)],cfg.print)
                    else:
                        for index_all in range(len(surf)):
                            if index_all not in decay_change_over:
                                cylinder_decay_array[index_all]= min(cylinder_decay_array[index_all] + cylinder_decay_once,max_decay)

                else:
                    print("retry:{}".format(now_num))
                all_results.append([error_sum,results])
                if success:
                    break

        except Exception as e:
            if now_num<len(cut_params)-1:
                fit_tolerance_array=np.ones((len(surf),)) *cut_params[now_num+1][2]
            cyl_decay = min(cylinder_decay_once * (now_num + 1), max_decay)
            cone_decay = min(cone_decay_once * (now_num + 1), max_decay)
            torus_decay = np.clip(np.asarray(torus_decay_once, dtype=np.float32) * (now_num + 1), None, max_decay)
            sphere_decay = np.clip(np.asarray(sphere_decay_once, dtype=np.float32) * (now_num + 1), None, max_decay)
            cylinder_decay_array = np.full((len(surf),), cyl_decay, dtype=np.float32)
            cone_decay_array = np.full((len(surf),), cone_decay, dtype=np.float32)
            torus_decay_array = np.ones((len(surf), 2), dtype=np.float32) * torus_decay
            sphere_decay_array = np.ones((len(surf), 2), dtype=np.float32) * sphere_decay
            print("error type:", type(e).__name__)
            print("error message:", e)
            print(traceback.print_exc(e))
            
    if use_display:
        if  mode==3 and len(all_results)>0: 
            all_results= sorted(all_results,key=lambda x:x[0],reverse=False)
            error_num, results = all_results[0]
            if error_num!=0: 
                results = sorted(results,key=lambda x:x[2],reverse=False) 
                for result in results:
                    all_res_faces.extend(result[0])
            display.EraseAll()
            for face in all_res_faces:
                display.DisplayShape(face, update=True,color=0,transparency=0.5)
            display.FitAll()
        start_display()


def main(args):

    ###################################################
    data_path = args.data_path
    surfs = np.load(data_path + "/"+"faces.npy")
    surfs = np.transpose(surfs, (0, 1, 3, 4, 2)) 
    surf_masks = np.load(data_path + "/"+ "faces_masks.npy")
    adj_matrices = np.load(data_path + "/"+ "adj_matrix.npy")
    
    start, end = args.start,args.end
    if start==None:start=0
    if end==None:end=len(surfs)
    os.makedirs(args.exp_path,exist_ok=True)

    face_workers = args.face_workers
    solid_workers = args.solid_workers
    use_fit_plane = args.use_fit_plane
    use_fit_cylinder = args.use_fit_cylinder
    use_fit_cone = args.use_fit_cone
    use_fit_torus = args.use_fit_torus
    use_fit_sphere = args.use_fit_sphere
    use_align = args.use_align
    use_final_fuse = args.use_final_fuse

    print("use_fit_plane:",use_fit_plane,type(use_fit_plane))
    print("use_fit_cylinder:",use_fit_cylinder,type(use_fit_cylinder))
    print("use_fit_cone:",use_fit_cone,type(use_fit_cone))
    print("use_fit_torus:",use_fit_torus,type(use_fit_torus))
    print("use_fit_sphere:",use_fit_sphere,type(use_fit_sphere))
    print("use_align:",use_align,type(use_align))
    print("use_final_fuse:",use_final_fuse,type(use_final_fuse))

    if args.recut_all == True:
        shutil.rmtree(args.data_path + "/" + "complete", ignore_errors=True)
    os.makedirs(args.data_path + "/" + "complete",exist_ok=True)
    length = len(surfs)
    all_indices = np.arange(length)
    start_end_mask = (all_indices >= start) & (all_indices < end)
    
    sucess_mask = np.zeros((length,),dtype=np.bool_)
    for fname in os.listdir(args.exp_path):
        if fname.endswith(".step") and fname.split("_")[0]==str(args.name):
            exist_index = int(fname.split(".")[0].split("_")[2])
            if 0 <= exist_index < length:
                sucess_mask[exist_index] = True
    
    over_mask = np.zeros((length,),dtype=np.bool_)
    for fname in os.listdir(args.data_path + "/" + "complete"):
        if fname.endswith(".txt"):
            exist_index = int(fname.split(".")[0])
            if 0 <= exist_index < length:
                over_mask[exist_index] = True

    all_mask = start_end_mask & ~sucess_mask & ~over_mask
    print("need cut num:",sum(all_mask))

    surfs = np.asarray(surfs)[all_mask]
    surf_masks = np.asarray(surf_masks)[all_mask]
    adj_matrices = np.asarray(adj_matrices)[all_mask]
    indice = all_indices[all_mask]
    save_name = [args.name] * len(indice)
    save_dir = [args.exp_path] * len(indice)
    data_paths = [data_path] * len(indice)

    # fewer faces in the front
    mask_sizes = surf_masks.sum(axis=1)
    order = np.argsort(mask_sizes)
    surfs = [surfs[i] for i in order]
    surf_masks = [surf_masks[i] for i in order]
    adj_matrices = [adj_matrices[i] for i in order]
    indice = indice[order]
    if solid_workers!=1:
        try:
            with ProcessPoolExecutor(max_workers=solid_workers,initializer=initializer) as executor: 
                futures = {
                        executor.submit(process_one_solid, 
                                        surfs[i], 
                                        surf_masks[i], 
                                        adj_matrices[i],
                                        indice[i], 
                                        save_name[i], 
                                        save_dir[i],
                                        data_paths[i],
                                        face_workers,
                                        use_fit_plane,
                                        use_fit_cylinder,
                                        use_fit_cone,
                                        use_fit_torus,
                                        use_fit_sphere,
                                        use_align,
                                        use_final_fuse
                                        ): i for i in range(len(surfs))
                }
                for future in tqdm(as_completed(futures), total=len(futures), disable=True):
                    try:
                        result = future.result(timeout=180)
                    except TimeoutError:
                        pass

        except Exception as e:
            print("error type:", type(e).__name__)
            print("error message:", e)
            # traceback.print_exc()
        gc.collect()
    else:
        for i in range(0,len(surfs)):
            process_one_solid(
                surfs[i],
                surf_masks[i],
                adj_matrices[i],
                i+start,
                args.name,
                args.exp_path,
                data_path,
                face_workers,
                use_fit_plane,
                use_fit_cylinder,
                use_fit_cone,
                use_fit_torus,
                use_fit_sphere,
                use_align,
                use_final_fuse
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--exp_path", type=str, default="./exp")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--name", type=int, default=0)
    parser.add_argument("--face_workers", type=int, default=8)
    parser.add_argument("--solid_workers", type=int, default=32)
    parser.add_argument("--use_fit_plane", type=str2bool, default=True)
    parser.add_argument("--use_fit_cylinder", type=str2bool, default=True)
    parser.add_argument("--use_fit_cone", type=str2bool, default=True)
    parser.add_argument("--use_fit_torus", type=str2bool, default=True)
    parser.add_argument("--use_fit_sphere", type=str2bool, default=True)
    parser.add_argument("--use_align", type=str2bool, default=True)
    parser.add_argument("--use_final_fuse", type=str2bool, default=False)
    parser.add_argument("--recut_all", type=str2bool, default=False)
    
    # start,end = cfg.num,cfg.num+1
    # args = parser.parse_args(["--start",str(start),"--end",str(end)])
    args = parser.parse_args()
    
    main(args)
    # python ./cut_faces/main.py --data_path ./gen_res --exp_path ./exp --start 0 --end 1 --name 0
    # python ./cut_faces/main.py --start 0 --end 1 --name 0 --use_align False
    
    
    


