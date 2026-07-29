import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import product,combinations

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool,BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge
from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS_Wire,topods_Face
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomLib import GeomLib_Tool
from OCC.Core.ShapeFix import ShapeFix_Wire

from cut_faces.utils_occ import are_vertice_same, cut_faces, fundamental_bop, fuse_faces,get_face_area,get_face_center, is_section


###################################################
def split_edge_from_points(edge,vertice,tolerance = 1e-3): 
    curve, u_min, u_max = BRep_Tool.Curve(edge)
    in_p = []
    for v in vertice:
        if BRepExtrema_DistShapeShape(v,edge).Value()<tolerance:
            p=BRep_Tool.Pnt(v)
            u_value = GeomLib_Tool.Parameter(curve, p, tolerance)
            in_p.append([u_value[1],p,v])
    in_p = sorted(in_p,key=lambda x:x[0])

    all_cutted_edges = []
    for i in range(len(in_p)-1):
        try:
            cutted_edge = BRepBuilderAPI_MakeEdge(curve, in_p[i][0], in_p[i+1][0]).Edge()
        except:
            cutted_edge = BRepBuilderAPI_MakeEdge(curve, in_p[i][2], in_p[i+1][2]).Edge()
        all_cutted_edges.append(cutted_edge)
    return all_cutted_edges


###################################################
def find_all_cirs(adj):
    G = nx.Graph()
    for u, neighbors in adj.items():
        for v in neighbors:
            G.add_edge(u, v)

    cycles = nx.minimum_cycle_basis(G)
    return cycles

def sort_edges(one_cir_edges,edge2pnt): #for builder
    sorted_edges_pnts=[[one_cir_edges[0],[edge2pnt[one_cir_edges[0]][0],edge2pnt[one_cir_edges[0]][1]]]]  
    one_cir_edges=one_cir_edges[1:]
    while one_cir_edges:
        for i in range(len(one_cir_edges)):
            now_edge=one_cir_edges[i]
            connect_num=None
            edge2pnt[now_edge][0]
            if edge2pnt[now_edge][0]==sorted_edges_pnts[-1][1][1]: connect_num = 0
            if edge2pnt[now_edge][1]==sorted_edges_pnts[-1][1][1]: connect_num = 1
            if connect_num!=None:
                now_edge_reverse = now_edge.Reversed()
                if connect_num == 1:
                    sorted_edges_pnts.append([now_edge_reverse,[edge2pnt[now_edge][connect_num],edge2pnt[now_edge][(connect_num+1)%2]]])
                else:
                    sorted_edges_pnts.append([now_edge,[edge2pnt[now_edge][connect_num],edge2pnt[now_edge][(connect_num+1)%2]]])
                one_cir_edges.remove(now_edge)
                break
 
    sorted_one_cir_edges=[e for e,[p1,p2] in sorted_edges_pnts]
    return sorted_one_cir_edges


def judge_face_surround(faces,tolerance = 1e-3): # for bool
    surround_arr = np.zeros((len(faces),len(faces)))
    for i in range(len(faces)):  
        for j in range(len(faces)):
            if i==j:continue
            v_j = get_face_center(faces[j])
            if BRepExtrema_DistShapeShape(v_j,faces[i]).Value()<tolerance :
                area_i = get_face_area(faces[i])
                area_j = get_face_area(faces[j])
                if area_j<area_i:
                    surround_arr[i,j]=1
    return surround_arr


def split_face_from_edges(face,edges):
    cirs2=[]
    pnt2code={}  # pnt:code
    edge2pnt={}  # edge:pnt
    G=defaultdict(list)
    for e in edges:
        curve, u_min, u_max = BRep_Tool.Curve(e)
        pnt1,pnt2 = gp_Pnt(),gp_Pnt()
        curve.D0(u_min,pnt1)
        curve.D0(u_max,pnt2)
        in_pnts=[None,None]
        for in_p in list(pnt2code.keys()): 
            if are_vertice_same(pnt1,in_p,tolerance=1e-4):
                in_pnts[0]=in_p
            if are_vertice_same(pnt2,in_p,tolerance=1e-4):
                in_pnts[1]=in_p
        if in_pnts[0]==None:
            in_pnts[0] = pnt1
            pnt2code.update({pnt1:len(pnt2code)+1})
        if in_pnts[1]==None:
            in_pnts[1] = pnt2
            pnt2code.update({pnt2:len(pnt2code)+1})
        
        # Pay attention to the two edge loop
        if pnt2code[in_pnts[1]] in G[pnt2code[in_pnts[0]]] and pnt2code[in_pnts[0]] in G[pnt2code[in_pnts[1]]]:
            if [pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]] not in cirs2 and [pnt2code[in_pnts[1]],pnt2code[in_pnts[0]]] not in cirs2:
                cirs2.append([pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]])
        else:
            G[pnt2code[in_pnts[0]]].append(pnt2code[in_pnts[1]])
            G[pnt2code[in_pnts[1]]].append(pnt2code[in_pnts[0]])
        edge2pnt.update({e:[in_pnts[0],in_pnts[1]]})

    # all loop
    cirs3 = find_all_cirs(G)
    cirs = [c for c in cirs3]
    cirs.extend(cirs2)

    # disjoint-set
    groups=defaultdict(set)  #{i,{}}
    for cir in cirs:
        connect_topo=set()
        for p in cir:
            for k,v in groups.items():
                if p in v: 
                    connect_topo.add(k)
                    break
        if len(connect_topo)==0:
            groups[len(groups)].update(cir)
        elif len(connect_topo)==1:
            groups[list(connect_topo)[0]].update(cir)
        else:
            need_topo = min(connect_topo)
            for topo in connect_topo:
                if topo!=need_topo:
                    groups[need_topo].update(groups[topo])
                    del groups[topo]
            groups[need_topo].update(cir)

   
    groups = sorted(groups.items(), key=lambda x: x[0])
    for i, data in enumerate(groups):
        groups[i]=[i,data[1]]
    groups = dict(groups)

    pnt_groups=[None for _ in range(len(groups))]    
    edges_groups=[None for _ in range(len(groups))]

    for i,group_code in groups.items():
        one_group_pnts=[]
        for p,c in pnt2code.items():
            if c in group_code:
                one_group_pnts.append(p)
        pnt_groups[i]=one_group_pnts

        one_group_edges = []
        for e,ps in edge2pnt.items():
            if ps[0] in one_group_pnts and ps[1] in one_group_pnts:
                one_group_edges.append(e)
        edges_groups[i]=one_group_edges
        

    uvmax_groups=[[999,999,-999,-999] for _ in range(len(groups))]  # umin,vmin,umax,vmax
    surf = BRep_Tool.Surface(face)
    sample_num=5
    shape_analysis = ShapeAnalysis_Surface(surf)
    for i in range(len(groups)):     
        for e in edges_groups[i]:  
            c, first, last = BRep_Tool.Curve(e)
            params = np.linspace(first, last,sample_num)
            for param in params:
                p = gp_Pnt()
                c.D0(param,p)
                uv = shape_analysis.ValueOfUV(p, 1e-4)
                if uv.X()<uvmax_groups[i][0]:
                    uvmax_groups[i][0] = uv.X()
                if uv.X()>uvmax_groups[i][2]:
                    uvmax_groups[i][2] = uv.X()
                if uv.Y()<uvmax_groups[i][1]:
                    uvmax_groups[i][1] = uv.Y()
                if uv.Y()>uvmax_groups[i][3]:
                    uvmax_groups[i][3] = uv.Y()

    
    inner_map=defaultdict(list) # {outgroup:[innergroup,...]}
    for i in range(len(groups)):
        inner_ratio=0
        inner_map[i] = []
        for j in range(len(groups)):
            if i==j:continue
            inner_count=0
            for p in pnt_groups[j]:
                uv = shape_analysis.ValueOfUV(p, 1e-4)
                if uv.X()>uvmax_groups[i][0] and uv.X()<uvmax_groups[i][2] and uv.Y()>uvmax_groups[i][1] and uv.Y()<uvmax_groups[i][3]:
                    inner_count+=1
            inner_ratio=inner_count/len(pnt_groups[j])
            if inner_ratio>=0.8: 
                inner_map[i].append(j) 
        
    faces_groups = defaultdict(list)
    faces_groups_code = defaultdict(list)
    for cir in cirs:
        #正常处理
        one_cir_pnts=[]
        for p,c in pnt2code.items():
            if c in cir:
                one_cir_pnts.append(p)
        one_cir_edges = []
        for e,ps in edge2pnt.items():
            if ps[0] in one_cir_pnts and ps[1] in one_cir_pnts:
                one_cir_edges.append(e)

        del_cir2_edges = []  
        split_cir2_edges = [] 
        for cir2 in cirs2:
            if set(cir2).issubset(set(cir)) : 
                cir2_pnts=[]
                for p,c in pnt2code.items():
                    if c in cir2: cir2_pnts.append(p)
                cir2_edges=[]
                for e,ps in edge2pnt.items():
                    if ps[0] in cir2_pnts and ps[1] in cir2_pnts:
                        cir2_edges.append(e)

                if set(cir2)!=set(cir):
                    del_cir2_edges.append(cir2_edges)

                elif set(cir2)==set(cir):
                    split_cir2_edges.append(cir2_edges)
    

        if del_cir2_edges!=[]:
            all_cir2_edges = [item for one_edges_list in del_cir2_edges for item in one_edges_list]
            part_one_cir_edges = [e for e in one_cir_edges if e not in all_cir2_edges]
            sample_edges_list = list(product(*del_cir2_edges))
            sample_face_params=[]
            for sample_edges in sample_edges_list:
                temp_part_one_cir_edges = [e for e in part_one_cir_edges]
                temp_part_one_cir_edges.extend(sample_edges)
                sorted_temp_part_one_cir_edges = sort_edges(temp_part_one_cir_edges,edge2pnt)
                wire = TopoDS_Wire()
                builder = BRep_Builder()
                builder.MakeWire(wire)
                for i in range(len(sorted_temp_part_one_cir_edges)):  builder.Add(wire,sorted_temp_part_one_cir_edges[i])
                fixer = ShapeFix_Wire(wire, face, 1e-3)
                fixer.FixConnected(1e-3)  
                fixer.Perform()
                surface = BRep_Tool.Surface(face)
                sample_face = BRepBuilderAPI_MakeFace(surface,wire).Face()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(sample_face, props) 
                sample_face_params.append([sample_edges,props.Mass()]) 
            need_edges=sorted(sample_face_params,key= lambda x:x[1])[0][0]
            for e in all_cir2_edges:one_cir_edges.remove(e)
            for e in need_edges:one_cir_edges.append(e)


        assert len(split_cir2_edges)<=1
        splitted_cir_edges=[]
        if split_cir2_edges!=[]: 
            split_cir2_edges = split_cir2_edges[0]
            combs=list(combinations(split_cir2_edges, 2))
            sample_face_params=[]
            for comb in combs:
                wire = TopoDS_Wire()
                builder = BRep_Builder()
                builder.MakeWire(wire)
                for i in range(len(comb)):  builder.Add(wire,comb[i])
                fixer = ShapeFix_Wire(wire, face, 1e-3)
                fixer.FixConnected(1e-3) 
                fixer.Perform()
                surface = BRep_Tool.Surface(face)
                sample_face = BRepBuilderAPI_MakeFace(surface,wire).Face()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(sample_face, props)  
                sample_face_params.append([list(comb),props.Mass()]) 
            need_edges_params=sorted(sample_face_params,key= lambda x:x[1])[:len(split_cir2_edges)-1]
            for es,m in need_edges_params:splitted_cir_edges.append(es)

        sorted_one_cir_edges = sort_edges(one_cir_edges,edge2pnt)
        wire = TopoDS_Wire()
        builder = BRep_Builder()
        builder.MakeWire(wire)
        for i in range(len(sorted_one_cir_edges)): builder.Add(wire,one_cir_edges[i])

        # fix
        fixer = ShapeFix_Wire(wire, face, 1e-3)
        fixer.FixConnected(1e-3)
        fixer.Perform()
        wire = fixer.Wire()
        surface = BRep_Tool.Surface(face)
        cutted_face = BRepBuilderAPI_MakeFace(surface,wire).Face()

        for i in range(len(groups)):
            if set(cir).issubset(set(groups[i])) and cutted_face is not None:
                faces_groups[i].append(cutted_face)
                faces_groups_code[i].append(cir) 
                break

    #group faces
    for i, faces_now_group in faces_groups.items():
        del_faces=[]
        add_faces=[]
        need_fused_faces=[]
        for j in inner_map[i]:
            need_fused_faces.extend(faces_groups[j])
        if len(need_fused_faces)==0: continue
        fused_face = fuse_faces(need_fused_faces)
        for f in faces_now_group:
            if is_section(f,fused_face,TopAbs_EDGE):
                new_f_compoound = fundamental_bop(f,fused_face,"cut")
                explorer = TopExp_Explorer(new_f_compoound, TopAbs_FACE)   #only one face
                while explorer.More(): 
                    new_f = topods_Face(explorer.Current())
                    explorer.Next()
                    add_faces.append(new_f)
                del_faces.append(f)
            
        for f in del_faces: faces_groups[i].remove(f)
        for f in add_faces: faces_groups[i].append(f)
    
    #bool operation
    del_idx=defaultdict(list)
    for group_idx, faces_now_group in faces_groups.items(): 
        surround_arr = judge_face_surround(faces_now_group)
        for i in range(len(surround_arr)):
            for j in range(len(surround_arr)): 
                if surround_arr[i,j]==1 and i!=j:
                    new_face = cut_faces(faces_now_group[i],[faces_now_group[j]])
                    if new_face!=None:
                        faces_now_group[i] = new_face
                    else:
                        del_idx[group_idx].append(i)
                        break
        
    return_faces=[]
    for group_idx,faces in faces_groups.items():
        for j in range(len(faces)):
            if j not in del_idx[group_idx]:
                return_faces.append(faces[j])
    return return_faces
