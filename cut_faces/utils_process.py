import numpy as np
from collections import defaultdict

from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face,TopoDS_Wire
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepLib import breplib_ExtendFace
from OCC.Core.AIS import AIS_Shape
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.Quantity import Quantity_Color,Quantity_NOC_RED
from OCC.Core.BRep import BRep_Tool,BRep_Builder
from OCC.Core.ShapeFix import ShapeFix_Wire

from cut_faces.utils_occ import get_face_uv_len,get_edges_from_face,are_edges_same, is_edge_in,get_bbox_norm,fundamental_bop, get_edge_inner_outer
from cut_faces.utils_fit import fit_plane,fit_cylinder,fit_cone,fit_torus,fit_sphere
from cut_faces.utils_split import sort_edges
from cut_faces.config import cfg

####################################### first postprocess
def fit_bspline_surface(
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
        cylinder_decay_array,
        cone_decay_array,
        torus_decay_array,
        sphere_decay_array,
        num_uv = 16
    ):

    ##########################################拟合
    fit_faces = [] 
    valid = sum(surf_mask)
    for i,points in enumerate(surf):
        DegMin,DegMax = 2,8
        face_type = "bspline"
        face_params = None
        tolerance = fit_tolerance_array[i]
        cylinder_decay = cylinder_decay_array[i] if use_cylinder_decay else 0
        cone_decay = cone_decay_array[i] if use_cone_decay else 0
        torus_decay = torus_decay_array[i] if use_torus_decay else [0,0]
        sphere_decay = sphere_decay_array[i] if use_sphere_decay else [0,0]
        
        if i>=valid:continue

        fit_data={}
        if use_fit_plane: 
            plane_points,plane_err,plane_sucess,plane_points_params = fit_plane(points)
            if plane_sucess: fit_data["plane"] = [plane_err,plane_points,plane_points_params]
        if use_fit_cylinder:
            cylinder_points,cylinder_err,cylinder_sucess,cylinder_points_params = fit_cylinder(points,decay=cylinder_decay)
            if cylinder_sucess: fit_data["cylinder"] = [cylinder_err,cylinder_points,cylinder_points_params]
        if use_fit_cone:
            cone_points,cone_err,cone_sucess,cone_points_params = fit_cone(points,decay=cone_decay)
            if cone_sucess: fit_data["cone"] = [cone_err,cone_points,cone_points_params]
        if use_fit_torus:
            torus_points,torus_err,torus_sucess,torus_points_params = fit_torus(points,decay=torus_decay)
            if torus_sucess: fit_data["torus"] = [torus_err,torus_points,torus_points_params]
        if use_fit_sphere:
            sphere_points,sphere_err,sphere_sucess,sphere_points_params = fit_sphere(points,decay=sphere_decay)
            if sphere_sucess: fit_data["sphere"] = [sphere_err,sphere_points,sphere_points_params]

        best_type, (best_err, best_points, best_params) = min(fit_data.items(),key=lambda kv: kv[1][0])

        if use_fit_plane and best_type == "plane" and best_err<0.025:
            points,DegMin,tolerance,fit_sucess,face_type,face_params = best_points,0,1e-3,True,"plane",best_params
            dprint([i,"plane"],cfg.print)
        elif use_fit_cylinder and best_type == "cylinder" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type,face_params = best_points,6,1e-3,True,"cylinder",best_params
            dprint([i,"cylinder",cylinder_decay],cfg.print)
        elif use_fit_cone and best_type == "cone" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type,face_params = best_points,6,1e-3,True,"cone",best_params
            dprint([i,"cone",cone_decay],cfg.print)
        elif use_fit_torus and best_type == "torus" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type,face_params = best_points,6,1e-3,True,"torus",best_params
            dprint([i,"torus",torus_decay],cfg.print)
        elif use_fit_sphere and best_type == "sphere" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type,face_params = best_points,6,1e-3,True,"sphere",best_params
            dprint([i,"sphere",sphere_decay],cfg.print)
            
        fit_faces.append(
            {
                "type": face_type,
                "points": points,
                "params": face_params,
                "err":best_err,
                "DegMin":DegMin,
                "DegMax":DegMax,
                "tolerance":tolerance,
            }
        )

    ###########################################construct bspline
    for i,fit_face in enumerate(fit_faces):
        points = fit_face["points"]
        uv_points_array = TColgp_Array2OfPnt(1, num_uv, 1, num_uv)
        for u_index in range(1, num_uv+1):
            for v_index in range(1, num_uv+1):
                pt = points[v_index-1, u_index-1]  
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, fit_face["DegMin"], fit_face["DegMax"], GeomAbs_C2, fit_face["tolerance"]).Surface()
        face = BRepBuilderAPI_MakeFace(approx_face,1e-8).Face()
        fit_faces[i].update({"surf":face})
    return fit_faces


def get_adj_faces_distance(faces,adj_matrix):
    max_distance={i:0 for i in range(len(faces))}
    for i in range(len(faces)):
        for j in range(len(faces)):
            if adj_matrix[i][j]==1:
                dis = BRepExtrema_DistShapeShape(faces[i]["surf"], faces[j]["surf"]).Value()
                if dis>max_distance[i]: max_distance[i] = dis
    return max_distance


def extend_faces_and_make_params(faces,max_distance,extend_length_dis,extend_length_const,edge_flags=None,extend_scales=None):
    face_map_params={}
    for i in range(len(faces)):
        flags = edge_flags[i] if edge_flags is not None and i < len(edge_flags) else [0,0,0,0]
        u_minus, u_plus, v_minus, v_plus = [bool(x) for x in flags]
        extend_scale = float(extend_scales[i]) if extend_scales is not None and i < len(extend_scales) else 1.0
        face = faces[i]["surf"]
        u_len, v_len = get_face_uv_len(face)
        modified_face=TopoDS_Face()
        extend_length = (max_distance[i]*extend_length_dis + extend_length_const)*(0.58+u_len/2)*extend_scale
        breplib_ExtendFace(face, extend_length, not u_minus, not u_plus, False, False, modified_face)
        extend_length = (max_distance[i]*extend_length_dis + extend_length_const)*(0.58+v_len/2)*extend_scale
        breplib_ExtendFace(modified_face, extend_length, False, False, not v_minus, not v_plus, modified_face)
        face_map_params.update({i:(face,modified_face)})
    return face_map_params

############################################ last postprocess
def del_surf_from_vert_degree_1(results,adj_matrix): #[[faces,error_flag,main_index],[...]]
    return_results = results
    for i in range(len(results)):
        main_faces = results[i][0]
        cut_faces = []
        for j in range(len(results)):
            if adj_matrix[i][j]==1:
                cut_faces.extend(results[j][0])

        for face in main_faces:
            edges, vertice = get_edges_from_face(face,return_edges_corners=True)
            del_flag = False
            for v in vertice:
                v_dis=[]
                for f in cut_faces:
                    dis  = BRepExtrema_DistShapeShape(v, f).Value()
                    v_dis.append(dis)
                if np.any(np.array(v_dis)<8e-3): # if up, del more
                    continue
                else:
                    del_flag = True
                    break
            if del_flag:
                dprint(["del1"],cfg.print)
                return_results[i][0].remove(face)
    return return_results


def del_surf_from_edge_degree_3(results): 
    def copy_list_list(data_list):
        new_list = [[],[],[],[]]
        for i,data in enumerate(data_list):
            for d in data:
                new_list[i].append(d)
        return new_list

    # data structure:
    # {main_index:[face1,face2...]}
    # {face:main_index}
    # {edge:[face1,face2...]}  
    # {face:[edge1,edge2...]}
    main_idx_to_face = defaultdict(list)
    face_to_main_idx = defaultdict(int)
    edge_to_face = defaultdict(list)
    face_to_edge = defaultdict(list)
    for i in range(len(results)):
        faces_one_surf = results[i][0]
        main_index = results[i][2]
        for face in faces_one_surf:
            edges = get_edges_from_face(face)
            main_idx_to_face[main_index].append(face)
            face_to_main_idx[face] = main_index
            for edge in edges:
                for edge_in in edge_to_face.keys():
                    if are_edges_same(edge,edge_in,sample_num=5,tolerance=1e-4):
                        edge_to_face[edge_in].append(face)
                        break
                else:
                    edge_to_face[edge].append(face)
    for edge,faces in edge_to_face.items():
        for face in faces:
            face_to_edge[face].append(edge)

    #[[face1,face2...],[edge1,edge2...],[d1_ori,d2_ori...],[d1_del,d2_del...]]
    del_data = [[],[],[],[]]
    end_flag = False
    while not end_flag: 
        for edge,faces in edge_to_face.items():
            get_del_face = False
            if len(set(faces)-set(del_data[0]))>=3:
                for face in faces: 
                    temp_del_data = copy_list_list(del_data) 
                    if face in temp_del_data[0]:continue
                    start_face_is_right=True 
                    if len(set(main_idx_to_face[face_to_main_idx[face]])-set(temp_del_data[0]))>=2: 
                        temp_del_data[0].append(face)
                        edges = face_to_edge[face]
                        for e in edges:
                            if e in temp_del_data[1]:
                                idx = temp_del_data[1].index(e)
                                temp_del_data[3][idx]+=1
                            else:
                                temp_del_data[1].append(e)
                                temp_del_data[2].append(len(edge_to_face[e]))
                                temp_del_data[3].append(1)

                        while np.any(np.array(temp_del_data[2]) - np.array(temp_del_data[3]) == 1): 
                            diff_list = list(np.array(temp_del_data[2]) - np.array(temp_del_data[3]))
                            edge_idx = diff_list.index(1)
                            attn_edge = temp_del_data[1][edge_idx]
                            attn_face = list(set(edge_to_face[attn_edge])-set(temp_del_data[0]))
                            if len(attn_face)!=1:
                                print("attn_face 2 err")
                                return
                            attn_face = attn_face[0]
                            if len(set(main_idx_to_face[face_to_main_idx[attn_face]])-set(temp_del_data[0]))>=2: 
                                temp_del_data[0].append(attn_face)
                                edges = face_to_edge[attn_face]
                                get_degree3 = False 
                                for e in edges:
                                    if e in temp_del_data[1]:
                                        idx = temp_del_data[1].index(e)
                                        temp_del_data[3][idx]+=1
                                    else:
                                        temp_del_data[1].append(e)
                                        temp_del_data[2].append(len(edge_to_face[e]))
                                        temp_del_data[3].append(1)
                                    
                                    if len(set(edge_to_face[e]) - set(temp_del_data[0]))+1>=3: #因为已经加进来了attn face，所以先忽略�?
                                        get_degree3=True

                                if not get_degree3: 
                                    start_face_is_right = False
                                    break
                            else: 
                                start_face_is_right = False
                                break

                    if start_face_is_right and del_data[2]!=temp_del_data[2]:
                        get_del_face = True
                        del_data = temp_del_data
                        break
                else:
                    dprint(["not del when appear 3edge"],cfg.print)

            if get_del_face:
                break            
        else:
            end_flag = True
            break

    # del face
    for f in del_data[0]:
        main_index = face_to_main_idx[f]
        results[main_index][0].remove(f)
        print("del3")
    return results



def final_fuse(faces,orign_face):
    if len(faces)==1:
        return faces[0]
    edge_select=[]
    for f in faces:  
        edges = get_edges_from_face(f)
        for e in edges:
            for e_in in edge_select:
                if are_edges_same(e,e_in,tolerance=1e-4):
                    edge_select.remove(e_in)
                    break
                if is_edge_in(e,e_in,tolerance=1e-4):
                    edge_select.remove(e_in)
                    new_e = fundamental_bop(e_in,e,op_name="cut")
                    edge_select.append(new_e)
                    break
                if is_edge_in(e_in,e,tolerance=1e-4):
                    edge_select.remove(e_in)
                    new_e = fundamental_bop(e,e_in,op_name="cut")
                    edge_select.append(new_e)
                    break
            else:
                edge_select.append(e)

    edges_groups,edge2pnt = get_edge_inner_outer(edge_select)
    wires = [] #[wire,bboxlen]
    for edges in edges_groups:
        sorted_one_cir_edges = sort_edges(edges,edge2pnt)
        wire = TopoDS_Wire()
        builder = BRep_Builder()
        builder.MakeWire(wire)
        for i in range(len(sorted_one_cir_edges)): 
            builder.Add(wire,sorted_one_cir_edges[i])
        fixer = ShapeFix_Wire(wire, orign_face, 1e-3)
        fixer.FixConnected(1e-3)
        fixer.Perform()
        wire = fixer.Wire()
        all_pnts = []
        for e in edges:
            for p in edge2pnt[e]:
                all_pnts.append(np.array([p.X(),p.Y(),p.Z()]))
        bboxlen = get_bbox_norm(np.array(all_pnts))
        wires.append([wire,bboxlen])

    wires = sorted(wires,key=lambda x: x[1],reverse=True)
    outer_wire_param = wires[0]
    inner_wires_param = wires[1:]
    # Cut by wires
    surface = BRep_Tool.Surface(orign_face)
    face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire_param[0])
    for wire in inner_wires_param:
        face_builder.Add(wire[0])
    final_face = face_builder.Shape()

    return final_face


############################################################## valid check
def check_brep_validity_by_edge(results):
    all_group_edges = []
    for result in results:
        for f in result[0]:
            edges = get_edges_from_face(f)
            for e in edges:
                all_group_edges.append(e)

    for i in range(len(all_group_edges)):
        for j in range(len(all_group_edges)):
            if i==j:continue
            ei,ej=all_group_edges[i],all_group_edges[j]
            if are_edges_same(ei,ej,tolerance=8e-2) or is_edge_in(ei,ej,tolerance=8e-2) or is_edge_in(ej,ei,tolerance=8e-2):
                break
        else:
            return False
    return True   

def check_brep_topo_validity(results):
    for result in results:
        if len(result[0])==0:
            return False
    return True


############################################################### other
def dprint(data,enable=True):
    if enable:
        print(data)


def show(shapes,color,type,display):
    if type=="face":
        for s in shapes:
            display.DisplayShape(s,color=color,transparency=0.5,update=True)
    elif type=="edge":
        for s in shapes: 
            ais = AIS_Shape(s)
            ais.SetColor(Quantity_Color(Quantity_NOC_RED))
            ais.SetWidth(6.0) 
            display.Context.Display(ais, True)
    elif type=="vertex":
        for s in shapes:
            pnt = BRep_Tool.Pnt(s)   
            sphere = BRepPrimAPI_MakeSphere(pnt, 0.04).Shape()
            display.DisplayShape(sphere,color='blue',update=False)