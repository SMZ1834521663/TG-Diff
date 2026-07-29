import os
import gc
import time
import argparse
import signal
import math
import json
import traceback
import threading
import subprocess
import platform
from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from collections import defaultdict
from itertools import product,combinations
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,TimeoutError,as_completed

if platform.system() == "Windows":
    from OCC.Display.SimpleGui import init_display

from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec, gp_Pnt2d, gp_Ax1, gp_Dir
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C2,GeomAbs_C1,GeomAbs_C0
from OCC.Core.BRep import BRep_Tool,BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeSolid,BRepBuilderAPI_Sewing
from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_VERTEX,TopAbs_WIRE,TopAbs_SHELL
from OCC.Core.TopoDS import TopoDS_Face,TopoDS_Edge,TopoDS_Vertex,TopoDS_Wire,TopoDS_Compound,TopoDS_Shell,TopoDS_Solid,topods_Face,topods_Edge,topods_Vertex,topods_Wire,topods_Shell
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common,BRepAlgoAPI_Section
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepLib import breplib_ExtendFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface,BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface,ShapeAnalysis_Wire,ShapeAnalysis_Shell,ShapeAnalysis_FreeBounds
from OCC.Core.GeomLib import GeomLib_Tool
from OCC.Core.ShapeFix import ShapeFix_Wire,ShapeFix_Face,ShapeFix_Edge
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BOPAlgo import BOPAlgo_Builder
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.BOPAlgo import BOPAlgo_BOP
from OCC.Core.BOPAlgo import BOPAlgo_Operation
from OCC.Core.GCPnts import GCPnts_AbscissaPoint

# display, start_display, add_menu, add_function_to_menu = init_display()

##################################################
def write_step_file(a_shape, filename, application_protocol="AP242DIS"): 
    # a few checks
    if a_shape.IsNull():
        raise AssertionError("Shape %s is null." % a_shape)
    if application_protocol not in ["AP203", "AP214IS", "AP242DIS"]:
        raise AssertionError("application_protocol must be either AP203 or AP214IS. You passed %s." % application_protocol)
    if os.path.isfile(filename):
        print("Warning: %s file already exists and will be replaced" % filename)
    # creates and initialise the step exporter
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", application_protocol)

    # transfer shapes and write file
    step_writer.Transfer(a_shape,STEPControl_AsIs) #,STEPControl_ManifoldSolidBrep
    status = step_writer.Write(filename)

    if not status == IFSelect_RetDone:
        raise IOError("Error while writing shape to STEP file.")
    if not os.path.isfile(filename):
        raise IOError("File %s was not saved to filesystem." % filename)
    
def write_stl_file(a_shape, filename, mode="ascii", linear_deflection=0.9, angular_deflection=0.5):
    """ export the shape to a STL file
    Be careful, the shape first need to be explicitely meshed using BRepMesh_IncrementalMesh
    a_shape: the topods_shape to export
    filename: the filename
    mode: optional, "ascii" by default. Can either be "binary"
    linear_deflection: optional, default to 0.001. Lower, more occurate mesh
    angular_deflection: optional, default to 0.5. Lower, more accurate_mesh
    """
    if a_shape.IsNull():
        raise AssertionError("Shape is null.")
    if mode not in ["ascii", "binary"]:
        raise AssertionError("mode should be either ascii or binary")
    if os.path.isfile(filename):
        print("Warning: %s file already exists and will be replaced" % filename)
    # first mesh the shape
    mesh = BRepMesh_IncrementalMesh(a_shape, linear_deflection, False, angular_deflection, True)
    #mesh.SetDeflection(0.05)
    mesh.Perform()
    if not mesh.IsDone():
        raise AssertionError("Mesh is not done.")

    stl_exporter = StlAPI_Writer()
    if mode == "ascii":
        stl_exporter.SetASCIIMode(True)
    else:  # binary, just set the ASCII flag to False
        stl_exporter.SetASCIIMode(False)
    stl_exporter.Write(a_shape, filename)

    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")

##################################################
def are_edges_same(edge1, edge2, sample_num = 5, tolerance=1e-6): 
    
    c1, first1, last1 = BRep_Tool.Curve(edge1)
    c2, first2, last2 = BRep_Tool.Curve(edge2)
    
    # curve1
    params = np.linspace(first1, last1, sample_num)
    for param in params:
        pnt = gp_Pnt()
        c1.D0(param,pnt)
        v=BRepBuilderAPI_MakeVertex(pnt).Vertex()
        dis = BRepExtrema_DistShapeShape(v,edge2).Value()
        if dis > tolerance:
            return False
    # curve2
    params = np.linspace(first2, last2, sample_num)
    for param in params:
        pnt = gp_Pnt()
        c2.D0(param,pnt)
        v=BRepBuilderAPI_MakeVertex(pnt).Vertex()
        dis = BRepExtrema_DistShapeShape(v,edge1).Value()
        if dis > tolerance:
            return False
    return True

def is_edge_in(edge_small, edge_large, sample_num = 5, tolerance=1e-5):
    c1, first1, last1 = BRep_Tool.Curve(edge_small)

    params = np.linspace(first1, last1, sample_num)
    for param in params:
        pnt1= gp_Pnt()
        c1.D0(param,pnt1)
        v1=BRepBuilderAPI_MakeVertex(pnt1).Vertex()
        dis = BRepExtrema_DistShapeShape(v1,edge_large).Value()
        if dis > tolerance:
            return False
    return True

def are_vertice_same(vertex1, vertex2, tolerance=1e-6):
    p1,p2 = vertex1,vertex2
    if not isinstance(vertex1,gp_Pnt): p1 = BRep_Tool.Pnt(vertex1)
    if not isinstance(vertex2,gp_Pnt): p2 = BRep_Tool.Pnt(vertex2)
    if p1.Distance(p2)>tolerance:
        return False
    return True

def is_section(shape1,shape2,section_type):
    section_compound = BRepAlgoAPI_Section(shape1, shape2).Shape()
    explorer = TopExp_Explorer(section_compound, section_type)
    section_count = 0
    while explorer.More(): 
        explorer.Next()
        section_count+=1
    if section_count>=1:
        return True
    else:
        return False

def is_edge_straight(edge, sample_num=5):
    curve, first, last = BRep_Tool.Curve(edge)

    # sample points
    params = np.linspace(first, last, sample_num)
    pts = np.array([curve.Value(u).Coord() for u in params])

    v0 = pts[-1] - pts[0]    
    if np.linalg.norm(v0) < 1e-8:
        return False,None
    v0 = v0 / np.linalg.norm(v0)

    for i in range(2, sample_num):
        vi = pts[i] - pts[0]
        if np.linalg.norm(vi) < 1e-8:
            continue

        if np.linalg.norm(np.cross(vi, v0)) > 1e-2:
            return False,None
    return True,v0

##################################################
def get_edges_from_face(face,remove_degree_1=False,return_edges_corners=False):
    vertex_degree = {}  #{topo_vertex,degree}
    edge_vertex={}  #{topo_edge,(topo_vertex*2)}
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())  
        edge_explorer.Next()

        temp_vertex=[]
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = topods_Vertex(vertex_explorer.Current())  
            vertex_explorer.Next()
            in_flag=False
            now_v=None
            for v_in,d_in in vertex_degree.items():
                if are_vertice_same(v_in,vertex):
                    in_flag=True
                    now_v = v_in
            if not in_flag:
                vertex_degree.update({vertex:1})
            else:
                vertex_degree[now_v]+=1
           
            temp_vertex.append(now_v if now_v else vertex)
          
        assert len(temp_vertex)==2
        edge_vertex.update({edge:temp_vertex})

    if remove_degree_1: 
        while 1 in vertex_degree.values():
            get_flag = False 
            for vertex,degree in vertex_degree.items():
                if degree==1:
                    for e,vs in edge_vertex.items():
                        for v in vs:
                            if are_vertice_same(vertex,v):
                                get_flag = True
                                break

                        if get_flag:
                            for v in vs:
                                vertex_degree[v]-=1
                                assert vertex_degree[v]>=0
                            del edge_vertex[e]
                            break

                if get_flag:
                    break
    if return_edges_corners:
        return list(edge_vertex.keys()), list(vertex_degree.keys())
    else:
        return list(edge_vertex.keys())
                    
def get_face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()

def get_face_center(face,return_vertex = True):
    surf_adaptor = BRepAdaptor_Surface(face)
    umin, umax = surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter()
    vmin, vmax = surf_adaptor.FirstVParameter(), surf_adaptor.LastVParameter()  
    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)
    surface = BRep_Tool.Surface(face)
    pnt = gp_Pnt()
    surface.D0(uc,vc,pnt)
    if return_vertex:
        v = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        return v
    else:
        return pnt
    
def get_vertice_from_edge(edge):
    vertice=[]
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods_Vertex(vertex_explorer.Current())  
        vertex_explorer.Next()
        vertice.append(vertex)
    return vertice    

def get_edge_inner_outer(edges):
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
        
        if pnt2code[in_pnts[1]] in G[pnt2code[in_pnts[0]]] and pnt2code[in_pnts[0]] in G[pnt2code[in_pnts[1]]]:
            if [pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]] not in cirs2 and [pnt2code[in_pnts[1]],pnt2code[in_pnts[0]]] not in cirs2:
                cirs2.append([pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]])
        else:
            G[pnt2code[in_pnts[0]]].append(pnt2code[in_pnts[1]])
            G[pnt2code[in_pnts[1]]].append(pnt2code[in_pnts[0]])
        edge2pnt.update({e:[in_pnts[0],in_pnts[1]]})

    groups=defaultdict(set)  
    for edge in edges:
        edge_pnt = edge2pnt[edge]
        edge_pnt_code = [pnt2code[edge_pnt[0]],pnt2code[edge_pnt[1]]]
        connect_topo=set()
        for p in edge_pnt_code:
            for k,v in groups.items():
                if p in v: 
                    connect_topo.add(k)
                    break
        if len(connect_topo)==0:
            groups[len(groups)].update(edge_pnt_code)
        elif len(connect_topo)==1:
            groups[list(connect_topo)[0]].update(edge_pnt_code)
        else:
            need_topo = min(connect_topo)
            for topo in connect_topo:
                if topo!=need_topo:
                    groups[need_topo].update(groups[topo])
                    del groups[topo]
            groups[need_topo].update(edge_pnt_code)

    groups = sorted(groups.items(), key=lambda x: x[0])
    for i, data in enumerate(groups):
        groups[i]=[i,data[1]]
    groups = dict(groups)

    return_edges = []
    for i, codes in groups.items():
        one_cir_edges = []
        for e,pnts in edge2pnt.items():
            if pnt2code[pnts[0]] in codes and pnt2code[pnts[1]] in codes:
                one_cir_edges.append(e)
        return_edges.append(one_cir_edges)
    return return_edges,edge2pnt

def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)

def get_edges_dis(edge1, edge2, sample_num=50):
    curve1, f1, l1 = BRep_Tool.Curve(edge1)
    curve2, f2, l2 = BRep_Tool.Curve(edge2)

    params1 = np.linspace(f1, l1, sample_num)
    params2 = np.linspace(f2, l2, sample_num)

    pts1 = np.array([curve1.Value(u).Coord() for u in params1])
    pts2 = np.array([curve2.Value(u).Coord() for u in params2])

    # shape: (sample_num, sample_num)
    diff = pts1[:,None,:] - pts2[None,:,:]
    dists = np.linalg.norm(diff, axis=2)

    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = pts1[i]
    p2 = pts2[j]
    vec = p2 - p1

    return dists.min(),vec

def get_face_uv_len(face, n_samples=50):
    surf = BRepAdaptor_Surface(face)

    u1, u2 = surf.FirstUParameter(), surf.LastUParameter()
    v1, v2 = surf.FirstVParameter(), surf.LastVParameter()

    v_mid = 0.5 * (v1 + v2)
    us = np.linspace(u1, u2, n_samples)
    pts_u = [surf.Value(u, v_mid) for u in us]
    u_len = sum(pts_u[i].Distance(pts_u[i+1]) for i in range(len(pts_u)-1))

    u_mid = 0.5 * (u1 + u2)
    vs = np.linspace(v1, v2, n_samples)
    pts_v = [surf.Value(u_mid, v) for v in vs]
    v_len = sum(pts_v[i].Distance(pts_v[i+1]) for i in range(len(pts_v)-1))

    return u_len, v_len

def get_edge_length(edge):
    adaptor = BRepAdaptor_Curve(edge)  
    first = adaptor.FirstParameter()
    last = adaptor.LastParameter()
    length = GCPnts_AbscissaPoint.Length(adaptor, first, last)
    return length

############################## 
def fundamental_bop(big, small, op_name):
    if op_name == 'cut':
        op = BRepAlgoAPI_Cut(big, small)
    elif op_name == 'fuse':
        op = BRepAlgoAPI_Fuse(big, small)
    elif op_name == 'common':
        op = BRepAlgoAPI_Common(big, small)
    op.SetFuzzyValue(1e-5)
    op.Build()
    return op.Shape()

def fuse_faces(faces):
    if len(faces)==1:
        return faces[0]
    
    builder = BOPAlgo_Builder()
    list_of_shapes = TopTools_ListOfShape()
    for face in faces:
        list_of_shapes.Append(face)
    builder.SetArguments(list_of_shapes)
    builder.SetRunParallel(True)
    builder.SetFuzzyValue(1e-6)
    builder.Perform()
    result_shape = builder.Shape()
    return result_shape

def cut_faces(face_ori, cut_faces):
    bop = BOPAlgo_BOP()
    object_list = TopTools_ListOfShape()
    object_list.Append(face_ori)
    tool_list = TopTools_ListOfShape()
    for face in cut_faces:
        tool_list.Append(face)
    
    bop.SetArguments(object_list)
    bop.SetTools(tool_list)
    bop.SetOperation(BOPAlgo_Operation.BOPAlgo_CUT)  
    bop.SetFuzzyValue(1e-6)
    bop.SetRunParallel(True)
    bop.Perform()
    result_shape = bop.Shape()

    explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
    if explorer.More():
        return topods_Face(explorer.Current())
    else:
        return None

def merge_edges(edges): #return topods edge
    bounded_curves = []
    for edge in edges:
        curve, first, last = BRep_Tool.Curve(edge)
        trimmed_curve = Geom_TrimmedCurve(curve, first, last)
        bounded_curves.append(trimmed_curve)

    merged_curve = GeomConvert_CompCurveToBSplineCurve(bounded_curves[0])
    for curve in bounded_curves[1:]:
        merged_curve.Add(curve, 1e-6)  
    new_edge = BRepBuilderAPI_MakeEdge(merged_curve.BSplineCurve()).Edge()
    return new_edge

###################################################
def remove_degenerate_edges(face):
    wire_maker = BRepBuilderAPI_MakeWire()
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    has_degenerated_edge = False
    while exp.More():
        edge = exp.Current()
        if not BRep_Tool.Degenerated(edge):
            wire_maker.Add(edge)
        else:
            has_degenerated_edge=True
        exp.Next()
    if not has_degenerated_edge:
        return face
    wire = wire_maker.Wire()
    surf = BRep_Tool.Surface(face)
    new_face = BRepBuilderAPI_MakeFace(surf, wire, True).Face()
    return new_face


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    fixer.FixAddNaturalBound()
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    face = remove_degenerate_edges(face)
    return face


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def sewing_cutted_faces(faces,tolerance = 1e-6):
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(tolerance)
    for face in faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape() #shell

    # Make a solid from the shell
    if isinstance(sewn_shell,TopoDS_Compound):
        print("result compound")
        return sewn_shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()
    print(type(solid))
    return solid

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


def find_all_cirs(adj):
    G = nx.Graph()
    for u, neighbors in adj.items():
        for v in neighbors:
            G.add_edge(u, v)

    cycles = nx.minimum_cycle_basis(G)
    return cycles


def sort_edges(one_cir_edges,edge2pnt): 
    sorted_edges_pnts=[[one_cir_edges[0],[edge2pnt[one_cir_edges[0]][0],edge2pnt[one_cir_edges[0]][1]]]]  # 边：{点:有没有接上} 从右边的点接上
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


def judge_face_surround(faces,tolerance = 1e-3): 
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
        

        if pnt2code[in_pnts[1]] in G[pnt2code[in_pnts[0]]] and pnt2code[in_pnts[0]] in G[pnt2code[in_pnts[1]]]:
            if [pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]] not in cirs2 and [pnt2code[in_pnts[1]],pnt2code[in_pnts[0]]] not in cirs2:
                cirs2.append([pnt2code[in_pnts[0]],pnt2code[in_pnts[1]]])
        else:
            G[pnt2code[in_pnts[0]]].append(pnt2code[in_pnts[1]])
            G[pnt2code[in_pnts[1]]].append(pnt2code[in_pnts[0]])
        edge2pnt.update({e:[in_pnts[0],in_pnts[1]]})


    cirs3 = find_all_cirs(G)
    cirs = [c for c in cirs3]
    cirs.extend(cirs2)


    groups=defaultdict(set) 
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

        fixer = ShapeFix_Wire(wire, face, 1e-3)
        fixer.FixConnected(1e-3)
        fixer.Perform()
        wire = fixer.Wire()

        surface = BRep_Tool.Surface(face)
        cutted_face = BRepBuilderAPI_MakeFace(surface,wire).Face()
        # shape_fix = ShapeFix_Face(cutted_face) 
        # shape_fix.Perform() 
        # cutted_face=shape_fix.Face()

        for i in range(len(groups)):
            if set(cir).issubset(set(groups[i])) and cutted_face is not None:
                faces_groups[i].append(cutted_face)
                faces_groups_code[i].append(cir)  
                break


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
    # print(len(return_faces))
    return return_faces


def del_hang_edge(edges,vertice,main_index):
    vertice_to_edges = defaultdict(list)
    for v in vertice:
        vertice_to_edges[v] = []

    for e in edges:
        v1,v2 = get_vertice_from_edge(e)
        get_v1,get_v2 = False,False
        for v in vertice:
            if are_vertice_same(v,v1,tolerance=1e-4):
                vertice_to_edges[v].append(e)
                get_v1 = True
            elif are_vertice_same(v,v2,tolerance=1e-4):
                vertice_to_edges[v].append(e)
                get_v2 = True
            if get_v1 and get_v2:
                break
        else:
            print(main_index,"no v1v2")
    
    for v,es in vertice_to_edges.items():
        if len(es)==1:
            if es[0] in edges:
                edges.remove(es[0])
            if v in vertice:
                vertice.remove(v)
    return edges,vertice


def detect_parallel_near_line(edge1,edge2): 
    e1_stright,v1 = is_edge_straight(edge1)
    e2_stright,v2 = is_edge_straight(edge2)
    if not e1_stright or not e2_stright:
        return False,None
    cosv = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(cosv)))
    if angle>10:
        return False,None
    dis,vec_1_to_2 = get_edges_dis(edge1,edge2)
    if dis>2e-1:
        return False,None
    return True ,vec_1_to_2


def cut_face_and_select(main_face_params,cut_faces_params,use_ef_section):  
    main_index = list(main_face_params.keys())[0]
    # if main_index!=5:return
    print("cutting:",main_index)
    


    edge_not_section_cut_index = []
    face_not_section_cut_index = [] 
    main_face_orign = list(main_face_params.values())[0][0]
    main_face_extended = list(main_face_params.values())[0][1]
    section_edges = []
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
            section_edges.append([e,cut_face_extended,cut_index])
        if section_count==0:
            face_not_section_cut_index.append(cut_index)

    # for e in section_edges: display.DisplayShape(e[0],color=0,transparency=False,update=True)
    # return 

    ############################################
    tol = 1e-4
    for now_num in range(4):
        section_points=[]
        for i in range(len(section_edges)):
            for j in range(i+1,len(section_edges)):
                section = BRepAlgoAPI_Section(section_edges[i][0], section_edges[j][0], False)
                section.SetFuzzyValue(tol)   ################################
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

    if use_ef_section or len(section_points)<len(section_edges):
        print("high per edge")
        section_points=[]
        for i in range(len(section_edges)):
            for j in range(i+1,len(section_edges)):
                section = BRepAlgoAPI_Section(section_edges[i][0], section_edges[j][1], False)
                section.SetFuzzyValue(1e-4)   ################################
                # section.Approximation(True)
                section.Build()
                section_point_compound = section.Shape()
                explorer = TopExp_Explorer(section_point_compound, TopAbs_VERTEX)
                while explorer.More(): 
                    section_points.append(topods_Vertex(explorer.Current()))
                    explorer.Next()

    for i in range(len(section_edges)):
        for j in range(i+1,len(section_edges)):
            if section_edges[i][2]==6 and section_edges[j][2] ==8 and main_index==3:
                pass
            is_parallel_near,vec_i_to_j = detect_parallel_near_line(section_edges[i][0],section_edges[j][0])
            if is_parallel_near:
                edge_not_section_cut_index.append([section_edges[i][2],section_edges[j][2],vec_i_to_j])

    # for v in section_points: display.DisplayShape(v,color=0,transparency=False,update=True)
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

    # for v in section_points_duplicated: display.DisplayShape(v,color=0,transparency=False,update=True)
    cutted_edges_all = []
    for e in section_edges:
        e =e[0]
        splitted_edges = split_edge_from_points(e,section_points_duplicated,tolerance=1e-3)
        cutted_edges_all.extend(splitted_edges)

    cutted_edges_all,section_points_duplicated = del_hang_edge(cutted_edges_all,section_points_duplicated,main_index)
    # for e in cutted_edges_all: display.DisplayShape(e,color=0,transparency=False,update=True)
    # return
    
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
    
    # for e in selected_edges: display.DisplayShape(e,color=0,transparency=False,update=True)
    # return

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
    # for e in selected_edges_duplicated: display.DisplayShape(e,color=0,transparency=False,update=True)
    # return

    ########################################
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
        print(main_index,"面error")
        error_flag = True
    

    return return_faces,error_flag,main_index,face_not_section_cut_index,edge_not_section_cut_index


######################################################## 
#################################################
def estimate_surface_points_curvature(points):

    num_u, num_v, _ = points.shape
    normals = np.zeros((num_u-1, num_v-1, 3))
    

    for i in range(num_u-1):
        for j in range(num_v-1):
            p0 = points[i,j]
            p1 = points[i+1,j]
            p2 = points[i,j+1]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-8:
                n /= norm
            normals[i,j] = n

    avg_u = np.nanmean(normals, axis=1) 
    avg_v = np.nanmean(normals, axis=0)

    avg_u /= np.linalg.norm(avg_u, axis=1, keepdims=True)
    avg_v /= np.linalg.norm(avg_v, axis=1, keepdims=True)


    dot_u = np.einsum('ijk,ik->ij', normals, avg_u)
    angle_u = np.degrees(np.arccos(np.clip(dot_u, -1, 1)))
    u_dir_angle_deg = np.nanmean(np.nanmax(angle_u, axis=1))


    dot_v = np.einsum('ijk,jk->ij', normals, avg_v)
    angle_v = np.degrees(np.arccos(np.clip(dot_v, -1, 1)))
    v_dir_angle_deg = np.nanmean(np.nanmax(angle_v, axis=0))


    return u_dir_angle_deg, v_dir_angle_deg

def estimate_surface_points_wrinkle(points, k=8, alpha=0.5, scale=2.0):
  
    H, W, _ = points.shape
    pts_flat = points.reshape(-1, 3)
    N = pts_flat.shape[0]

    tree = cKDTree(pts_flat)
    dists, idxs = tree.query(pts_flat, k=k)

    C_raw = np.zeros(N, dtype=float)
    for i in range(N):
        neigh = pts_flat[idxs[i]]        # k x 3
        mu = neigh.mean(axis=0)
        X = neigh - mu
        Cov = (X.T @ X) / (X.shape[0] + 1e-12)
        w, _ = np.linalg.eigh(Cov)
        s = w.sum()
        if s > 0:
            C_raw[i] = w[0] / s
        else:
            C_raw[i] = 0.0


    C_scaled = (C_raw ** alpha) * scale


    wrinkle_score = C_scaled.mean()  

    return wrinkle_score

def estimate_surface_points_uvlen(points):
    u_len = 0
    for i in range(points.shape[1]): 
        pts = points[:,i,:]
        u_len += np.sum(np.linalg.norm(pts[1:,:] - pts[:-1,:], axis=1))
    u_len /= points.shape[1] 

    v_len = 0
    for i in range(points.shape[0]):  
        pts = points[i,:,:]
        v_len += np.sum(np.linalg.norm(pts[1:,:] - pts[:-1,:], axis=1))
    v_len /= points.shape[0]

    return u_len,v_len

def construct_laplacian_uv_weights(num_uv, alpha=0.3):
    grid = np.indices((num_uv, num_uv))
    di = np.minimum(grid[0], num_uv-1-grid[0])
    dj = np.minimum(grid[1], num_uv-1-grid[1])
    d = np.minimum(di, dj)  
    W = np.exp(-alpha * d)  
    return W

def construct_laplacian_adj(num_u, num_v, connectivity=4):

    N = num_u * num_v
    adj = np.zeros((N, N), dtype=np.float32)

    def idx(i, j):
        return i * num_v + j

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1,-1), (-1,1), (1,-1), (1,1)]

    for i in range(num_u):
        for j in range(num_v):
            cur_idx = idx(i,j)
            for du, dv in offsets:
                ni, nj = i+du, j+dv
                if 0 <= ni < num_u and 0 <= nj < num_v:
                    adj[cur_idx, idx(ni,nj)] = 1.0
    return adj

def laplacian_smooth_points(surf, iters=1, lambd=0.0, W=None):

    adj = construct_laplacian_adj(16,16)
    U, V, _ = surf.shape
    N = U * V
    L = np.diag(adj.sum(axis=1)) - adj  
    W = construct_laplacian_uv_weights(16,alpha=0.8).flatten()

    new_surf = surf.copy()
    new_lambd = lambd

    #
    u_deg,v_deg = estimate_surface_points_curvature(new_surf)
    #
    wrinkle_deg = estimate_surface_points_wrinkle(new_surf)
    #
    u_len,v_len = estimate_surface_points_uvlen(new_surf)

    # u_len,v_len = v_len,u_len
    u_deg,v_deg = v_deg,u_deg
    #################
    if (u_deg>40 and u_len<0.2) or (v_deg>40 and v_len<0.2): 
        if u_deg>v_deg:
            new_lambd = 0.15 * np.clip(v_len/u_len,0.6,2)
        elif u_deg<v_deg:
            new_lambd = 0.15 * np.clip(u_len/v_len,0.6,2)
    elif (u_deg>40 and u_len<0.4) or (v_deg>40 and v_len<0.4): 
        if u_deg>v_deg:
            new_lambd = 0.12 * np.clip(v_len/u_len,0.6,2)
        elif u_deg<v_deg:
            new_lambd = 0.12 * np.clip(u_len/v_len,0.6,2)
    elif (u_deg>40 and u_len<0.6) or (v_deg>40 and v_len<0.6): 
        if u_deg>v_deg:
            new_lambd = 0.10 * np.clip(v_len/u_len,0.6,2)
        elif u_deg<v_deg:
            new_lambd = 0.10 * np.clip(u_len/v_len,0.6,2)
    else:
        if u_deg>v_deg:
            new_lambd = 0.08 * np.clip(v_len/u_len,0.6,2)
        elif u_deg<v_deg:
            new_lambd = 0.08 * np.clip(u_len/v_len,0.6,2)
    if u_deg<15 and v_deg<15:#
        if wrinkle_deg<0.01 : 
            new_lambd=0.15
        elif wrinkle_deg>0.01 and wrinkle_deg<=0.02: 
            new_lambd=0.27
        elif wrinkle_deg<0.03 and wrinkle_deg>0.02:
            new_lambd=0.36
        elif wrinkle_deg<0.04 and wrinkle_deg>0.03:
            new_lambd=0.42
            
    #
    P_flat = new_surf.reshape(N, 3)
    for _ in range(iters):
        LP = L @ P_flat
        P_flat = P_flat - (new_lambd * W[:, None]) * LP
    new_surf = P_flat.reshape(U, V, 3)
    return new_surf


###########################
def fit_plane(points):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = H, W
    err_list=[]
    try:
        # ---------------------------
        centroid = np.mean(pts, axis=0, keepdims=True)
        X = pts - centroid
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        _, _, normal = Vh
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        # ---------------------------
        du = points[:, 1:, :] - points[:, :-1, :]
        du = du.reshape(-1, 3)
        u_dir = np.mean(du, axis=0)

        dv = points[1:, :, :] - points[:-1, :, :]
        dv = dv.reshape(-1, 3)
        v_dir = np.mean(dv, axis=0)

        u_dir = u_dir - np.dot(u_dir, normal) * normal
        v_dir = v_dir - np.dot(v_dir, normal) * normal

        u_dir = u_dir / (np.linalg.norm(u_dir) + 1e-8)
        v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-8)

        # ---------------------------
        vec = pts - centroid
        dist = np.dot(vec, normal.reshape(3,1))
        proj3d = pts - dist * normal.reshape(1,3)

        u = np.dot(proj3d - centroid, u_dir)
        v = np.dot(proj3d - centroid, v_dir)
        uv = np.stack([u, v], axis=1)
        umin, umax = uv[:, 0].min(), uv[:, 0].max()
        vmin, vmax = uv[:, 1].min(), uv[:, 1].max()

        # ---------------------------

        u_vals = np.linspace(umin, umax, n_u)
        v_vals = np.linspace(vmin, vmax, n_v)
        uu, vv = np.meshgrid(u_vals, v_vals)
        uu = uu.flatten()
        vv = vv.flatten()
        grid_points = centroid + uu[:, None] * u_dir + vv[:, None] * v_dir
        sampled_points = grid_points.reshape(H, W, 3)

        sucess = True
        err = evaluate_fit_error(points,sampled_points)
        err_list.append([err,sampled_points,sucess])

    except Exception as e:
        pass
        # print(traceback.print_exc(e))

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sucess = return_data
        return err,sampled_points,sucess
    else:
        return 999,None,False
    

def fit_cylinder(points,decay=0):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u, n_v = H,W
    err_list=[]

    # ---------------------------

    mcd = MinCovDet(store_precision=True, 
                    assume_centered=False, 
                    support_fraction=None, 
                    random_state=1).fit(pts)
    cov = mcd.covariance_
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dirs = eigvecs

    col_dirs = points[-1, :, :] - points[0, :, :]
    axis_dir_new1 = np.mean(col_dirs, axis=0)
    axis_dir_new1 /= np.linalg.norm(axis_dir_new1)
    col_dirs = np.transpose(points,(1,0,2))[-1, :, :] - np.transpose(points,(1,0,2))[0, :, :]
    axis_dir_new2 = np.mean(col_dirs, axis=0)
    axis_dir_new2 /= np.linalg.norm(axis_dir_new2)
    axis_dirs = np.vstack([axis_dirs,axis_dir_new1.reshape(1,3),axis_dir_new2.reshape(1,3)])

    for i, axis_dir in enumerate(axis_dirs):
        try:
            # ---------------------------

            pts = points.reshape(-1, 3)
            proj = np.dot(pts, axis_dir)
            min_proj, max_proj = proj.min(), proj.max()
            height = max_proj - min_proj

            atols = [0.05*height,0.1*height,0.15*height,0.8*height]
            for atol in atols:
                try:
                    base_mask = np.isclose(proj, min_proj, atol=atol)
                    base_points = pts[base_mask]
                    base_center = base_points.mean(axis=0)

                    # ---------------------------

                    ref = np.array([0, 0, 1])
                    if np.allclose(np.abs(np.dot(axis_dir, ref)), 1.0, atol=1e-3):
                        ref = np.array([1, 0, 0])
                    u = np.cross(axis_dir, ref)
                    u /= np.linalg.norm(u)
                    v = np.cross(axis_dir, u)

                    # ---------------------------
      
                    pts_2d = np.dot(base_points - base_center, np.c_[u, v])
                    x, y = pts_2d[:, 0], pts_2d[:, 1]
                    A = np.c_[2*x, 2*y, np.ones_like(x)]
                    b = x**2 + y**2
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    cx, cy, c3 = sol
                    radius = np.sqrt(c3 + cx**2 + cy**2)
                    circle_center_3d = base_center + cx*u + cy*v

                    # ---------------------------
            
                    angles = np.arctan2(pts_2d[:, 1] - cy, pts_2d[:, 0] - cx)
                    angles = np.unwrap(angles)
                    angle_min, angle_max = np.min(angles), np.max(angles)
                    sucess = False
                    if angle_max-angle_min>np.pi/8 and angle_max-angle_min<np.pi/0.5:
                        sucess = True

                    decay_value = decay*(angle_max-angle_min)
                    angle_min_decay,angle_max_decay = angle_min+decay_value,angle_max-decay_value

                    # ---------------------------
             
                    u_angles = np.linspace(angle_min, angle_max, n_u)
                    v_heights = np.linspace(0, height, n_v)
                    U, Vh = np.meshgrid(u_angles, v_heights)
                    sampled_points = (circle_center_3d[None,None,:]
                                    + radius*np.cos(U[:,:,None])*u[None,None,:]
                                    + radius*np.sin(U[:,:,None])*v[None,None,:]
                                    + Vh[:,:,None]*axis_dir[None,None,:])
                    # ---------------------------
           
                    u_angles = np.linspace(angle_min_decay, angle_max_decay, n_u)
                    v_heights = np.linspace(0, height, n_v)
                    U, Vh = np.meshgrid(u_angles, v_heights)
                    sampled_points_decay = (circle_center_3d[None,None,:]
                                    + radius*np.cos(U[:,:,None])*u[None,None,:]
                                    + radius*np.sin(U[:,:,None])*v[None,None,:]
                                    + Vh[:,:,None]*axis_dir[None,None,:])

                    err = evaluate_fit_error(points,sampled_points)
                    err_list.append([err,sampled_points,sampled_points_decay,sucess])

                except Exception as e:
                    pass
                    # print(traceback.print_exc(e))
        except Exception as e:
            pass
            # print(traceback.print_exc(e))

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sampled_points_decay,sucess = return_data
        return err,sampled_points_decay,sucess
    else:
        return 999,None,False


def fit_cone(points,decay=0.0):
    H, W, _ = points.shape
    pts = points.reshape(-1,3)
    n_u, n_v = H, W
    err_list = []
    candidates_points = [points, np.transpose(points,(1,0,2))]
    for cand_points in candidates_points:
        try:
            # ---------------------------
  
            local_centers = []
            local_radii = []

            for i in range(H):
                row_pts = cand_points[i,:,:]
               
                centroid = row_pts.mean(axis=0)
                U, S, Vt = np.linalg.svd(row_pts - centroid)
                normal = Vt[-1]  

              
                u = Vt[0]
                v = Vt[1]

             
                pts2d = (row_pts - centroid) @ np.c_[u, v]
                x, y = pts2d[:,0], pts2d[:,1]

                A = np.c_[2*x, 2*y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, c3 = np.linalg.lstsq(A,b,rcond=None)[0]
                    center_2d = np.array([cx, cy])
                    center_3d = centroid + cx*u + cy*v
                    local_centers.append(center_3d)
                    # 
                    radii = np.linalg.norm(row_pts - center_3d, axis=1)
                    local_radii.append(radii.mean())
                except:
                    continue

            local_centers = np.array(local_centers)
            local_radii = np.array(local_radii)

            # ---------------------------
     
            # ---------------------------
            base_center = local_centers.mean(axis=0)
            U, S, Vt = np.linalg.svd(local_centers - base_center)
            axis_dir = Vt[0]
            axis_dir /= np.linalg.norm(axis_dir)

            # ---------------------------
         
            s_all = (pts - base_center) @ axis_dir
            s_min, s_max = s_all.min(), s_all.max()
            height = s_max - s_min

            # ---------------------------
      
            r = np.linalg.norm(pts - (base_center + np.outer(s_all, axis_dir)), axis=1)
            k, c = np.linalg.lstsq(np.vstack([s_all, np.ones_like(s_all)]).T, r, rcond=None)[0]
            if k <= 0:
                axis_dir = -axis_dir
                s_all = (pts - base_center) @ axis_dir
                k, c = np.linalg.lstsq(np.vstack([s_all, np.ones_like(s_all)]).T, r, rcond=None)[0]

            # ---------------------------

        
            ref = np.array([0,0,1])
            if np.allclose(abs(np.dot(axis_dir, ref)), 1.0):
                ref = np.array([1,0,0])
            u = np.cross(axis_dir, ref)
            u /= np.linalg.norm(u)
            v = np.cross(axis_dir, u)

           
            atols = [0.1*height,0.05*height,0.15*height,0.8*height]
            for atol in atols:
                try:
                    base_mask = np.isclose(s_all, s_max, atol=atol)
                    base_pts = pts[base_mask]
                    pts2d = (base_pts - base_center) @ np.c_[u,v]
                    angles = np.arctan2(pts2d[:,1], pts2d[:,0])
                    angles = np.unwrap(angles)
                    angle_min, angle_max = angles.min(), angles.max()
                    sucess = False
                    if angle_max-angle_min>np.pi/8 and angle_max-angle_min<np.pi/0.5:
                        sucess = True

                  
                    delta_r = abs(k * height) / r.mean()
                    if abs(k) < 0.05 or delta_r < 0.03:
                        sucess = False

                    decay_value = decay*(angle_max-angle_min)
                    angle_min_decay,angle_max_decay = angle_min+decay_value,angle_max-decay_value
                    # ---------------------------
           
                    U_grid, V_grid = np.meshgrid(np.linspace(angle_min, angle_max, n_u),np.linspace(s_min, s_max, n_v))
                    radii = k*V_grid + c
                    radii = np.maximum(radii, 1e-6)
                    sampled_points = base_center[None,None,:]\
                                    + V_grid[:,:,None]*axis_dir[None,None,:] \
                                    + radii[:,:,None]*(np.cos(U_grid[:,:,None])*u[None,None,:]+ np.sin(U_grid[:,:,None])*v[None,None,:])
                    
                    # ---------------------------
               
          
                    U_grid, V_grid = np.meshgrid(np.linspace(angle_min_decay, angle_max_decay, n_u),np.linspace(s_min, s_max, n_v))
                    radii = k*V_grid + c
                    radii = np.maximum(radii, 1e-6)
                    sampled_points_decay = base_center[None,None,:]\
                                    + V_grid[:,:,None]*axis_dir[None,None,:] \
                                    + radii[:,:,None]*(np.cos(U_grid[:,:,None])*u[None,None,:]+ np.sin(U_grid[:,:,None])*v[None,None,:])

                    err = evaluate_fit_error(pts,sampled_points)
                    err_list.append([err,sampled_points,sampled_points_decay,sucess])
                except Exception as e:
                    pass
                    # print(traceback.print_exc(e))
        except Exception as e:
            pass
            # print(traceback.print_exc(e))

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sampled_points_decay,sucess = return_data
        return err,sampled_points_decay,sucess
    else:
        return 999,None,False

def fit_torus(points,decay=[0.00,0.00]):
    H, W, _ = points.shape
    pts = points.reshape(-1,3)
    n_u,n_v = W,H
    err_list = []
    candidates_points = [points, np.transpose(points,(1,0,2))]
    for cand_points in candidates_points:
        try:
            # -------------------------------
   
            small_centers = []
            small_radii = []
            for i in range(W):
                col_pts = cand_points[:,i,:]
                centroid = col_pts.mean(axis=0)
                U_, S_, Vt_ = np.linalg.svd(col_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (col_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:,0], pts2d[:,1]
                A = np.c_[2*x, 2*y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A,b,rcond=None)[0]
                    center_3d = centroid + cx*u_plane + cy*v_plane
                    small_centers.append(center_3d)
                    r_local = np.mean(np.sqrt(np.sum((col_pts - center_3d)**2, axis=1)))
                    small_radii.append(r_local)
                except:
                    continue
            small_centers = np.array(small_centers)
            r = np.mean(small_radii)

            # -------------------------------
       
            plane_center = small_centers.mean(axis=0)
            _,_,Vt = np.linalg.svd(small_centers - plane_center)
            plane_normal = Vt[-1]  

            # -------------------------------
    
            local_centers = []
            for i in range(H):
                row_pts = cand_points[i,:,:]
                centroid = row_pts.mean(axis=0)
                U_, S_, Vt_ = np.linalg.svd(row_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (row_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:,0], pts2d[:,1]
                A = np.c_[2*x, 2*y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A,b,rcond=None)[0]
                    center_3d = centroid + cx*u_plane + cy*v_plane
                    local_centers.append(center_3d)
                except:
                    continue
            local_centers = np.array(local_centers)
            base_center = local_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(local_centers - base_center)
            axis_dir = Vt[0]
            axis_dir /= np.linalg.norm(axis_dir)

            # -------------------------------
   
            t = np.dot(plane_center - base_center, plane_normal) / (np.dot(axis_dir, plane_normal)+1e-12)
            center = base_center + t*axis_dir

            # -------------------------------

            ref = np.array([1,0,0])
            if abs(np.dot(ref, axis_dir)) > 0.9:
                ref = np.array([0,1,0])
            u_vec = np.cross(axis_dir, ref)
            u_vec /= np.linalg.norm(u_vec)
            v_vec = np.cross(axis_dir, u_vec)


            X = (small_centers - center) @ u_vec
            Y = (small_centers - center) @ v_vec
            u_angle = np.arctan2(Y, X)
            u_angle = np.unwrap(u_angle)
            u_min, u_max = u_angle.min(), u_angle.max()


            v_angles = []

            for i in range(W):
                col_pts = cand_points[:,i,:]
                small_center = small_centers[i]
                pts_vecs = col_pts - small_center 
                
            
                radial_vec = small_center - center
                radial_vec /= np.linalg.norm(radial_vec)
                radial_proj = np.dot(pts_vecs, radial_vec)
                
     
                axis_proj = np.dot(pts_vecs, axis_dir)
                angles = np.arctan2(axis_proj, radial_proj)
                angles_unwrapped = np.unwrap(angles)

                if i > 0:
                    prev = v_angles[-1]
                    cur_min, cur_max = angles_unwrapped.min(), angles_unwrapped.max()
                    prev_min, prev_max = prev.min(), prev.max()
                    if cur_max < prev_min:
                        angles_unwrapped += 2*np.pi
                    elif cur_min > prev_max:
                        angles_unwrapped -= 2*np.pi

                v_angles.append(angles_unwrapped)    
            
            mins = np.array([col.min() for col in v_angles])
            maxs = np.array([col.max() for col in v_angles])
            v_min = mins.mean()
            v_max = maxs.mean()

            sucess = False
            if u_max-u_min>np.pi/8 and v_max-v_min>np.pi/8:
                sucess = True
            # print(sucess)
            decay_value = decay[0]*(u_max-u_min)
            u_min_decay,u_max_decay = u_min+decay_value,u_max-decay_value

            decay_value = decay[1]*(v_max-v_min)
            v_min_decay,v_max_decay = v_min+decay_value,v_max-decay_value

            # -------------------------------
  
            R = np.mean(np.linalg.norm(small_centers - center, axis=1))

            # -------------------------------
    
            full_circle = np.isclose(u_max-u_min, 2*np.pi)
            u = np.linspace(u_min, u_max, n_u, endpoint=not full_circle)
            v = np.linspace(v_min, v_max, n_v)  
            U, V = np.meshgrid(u, v)
            X = (R + r*np.cos(V)) * np.cos(U)
            Y = (R + r*np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
            sampled_local = np.stack([X, Y, Z], axis=2)
            pts_local = sampled_local.reshape(-1,3)  
            R_mat = np.stack([u_vec, v_vec, axis_dir], axis=1)
            sampled_points = (R_mat @ pts_local.T).T + center
            sampled_points = sampled_points.reshape(n_v, n_u, 3)

            # -------------------------------
     
            full_circle = np.isclose(u_max_decay-u_min_decay, 2*np.pi)
            u = np.linspace(u_min_decay, u_max_decay, n_u, endpoint=not full_circle)
            v = np.linspace(v_min_decay, v_max_decay, n_v) 
            U, V = np.meshgrid(u, v)
            X = (R + r*np.cos(V)) * np.cos(U)
            Y = (R + r*np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
            sampled_local = np.stack([X, Y, Z], axis=2)
            pts_local = sampled_local.reshape(-1,3)   
            R_mat = np.stack([u_vec, v_vec, axis_dir], axis=1)
            sampled_points_decay = (R_mat @ pts_local.T).T + center
            sampled_points_decay = sampled_points_decay.reshape(n_v, n_u, 3)

            err = evaluate_fit_error(pts,sampled_points)
            err_list.append([err,sampled_points,sampled_points_decay,sucess])

        except Exception as e:
            pass
            # print(traceback.print_exc(e))

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sampled_points_decay,sucess = return_data
        return err,sampled_points_decay,sucess
    else:
        return 999,None,False


def fit_sphere(points, decay=[0.0,0.0]):
    H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    n_u,n_v = W,H
    err_list = []
    candidates_points = [points, np.transpose(points,(1,0,2))]
    for cand_points in candidates_points:
        try:
            # -------------------------------
     
            X, Y, Z = pts[:,0], pts[:,1], pts[:,2]
            A = np.c_[2*X, 2*Y, 2*Z, np.ones_like(X)]
            b = X**2 + Y**2 + Z**2
            cx, cy, cz, D = np.linalg.lstsq(A, b, rcond=None)[0]
            center = np.array([cx, cy, cz])
            radius = np.sqrt(D + cx**2 + cy**2 + cz**2)
            
            # -------------------------------

            local_centers = []
            for i in range(H):
                row_pts = cand_points[i,:,:]
                centroid = row_pts.mean(axis=0)
                U_, S_, Vt_ = np.linalg.svd(row_pts - centroid)
                u_plane, v_plane = Vt_[0], Vt_[1]
                pts2d = (row_pts - centroid) @ np.c_[u_plane, v_plane]
                x, y = pts2d[:,0], pts2d[:,1]
                A = np.c_[2*x, 2*y, np.ones_like(x)]
                b = x**2 + y**2
                try:
                    cx, cy, _ = np.linalg.lstsq(A,b,rcond=None)[0]
                    center_3d = centroid + cx*u_plane + cy*v_plane
                    local_centers.append(center_3d)
                except:
                    continue
            local_centers = np.array(local_centers)
            base_center = local_centers.mean(axis=0)
            _, _, Vt = np.linalg.svd(local_centers - base_center)
            z_axis = Vt[0]
            z_axis /= np.linalg.norm(z_axis)

            # -------------------------------
  
      
            ref = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(ref, z_axis)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            
            x_axis = np.cross(ref, z_axis)
            if np.linalg.norm(x_axis) < 1e-10: 
                ref = np.array([0.0, 0.0, 1.0])
                x_axis = np.cross(ref, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            
            first_vec = cand_points[0,0,:] - center
            if np.dot(first_vec, x_axis) < 0:
                x_axis = -x_axis
                y_axis = np.cross(z_axis, x_axis)

            R = np.column_stack([x_axis, y_axis, z_axis])

            # -------------------------------
       
            vecs = cand_points - center 
            r_len = np.linalg.norm(vecs, axis=2)
            
            x_l = vecs @ x_axis 
            y_l = vecs @ y_axis
            z_l = vecs @ z_axis
   
            rows = [cand_points[H//2 - 1], cand_points[H//2], cand_points[H//2 + 1]]
            u_min_list, u_max_list = [], []
            for row in rows:
                vecs = row - center
                x = vecs @ x_axis
                y = vecs @ y_axis
                ang = np.arctan2(y, x)
                ang_s = np.sort(ang)
                diff = np.diff(np.r_[ang_s, ang_s[0] + 2*np.pi])
                k = np.argmax(diff)
                umin = ang_s[(k + 1) % len(ang_s)]
                umax = ang_s[k]
                if umax < umin:
                    umax += 2*np.pi
                u_min_list.append(umin)
                u_max_list.append(umax)

            u_min = np.mean(u_min_list)
            u_max = np.mean(u_max_list)
            if u_max < u_min:
                u_max += 2*np.pi

            v_angle = np.arccos(z_l / r_len)   
            v_min,v_max = v_angle.min(), v_angle.max()

            sucess = False
            if u_max-u_min>np.pi/6 and v_max-v_min>np.pi/6 and 1/3<((u_max-u_min)/((v_max-v_min)*2))<3:
                sucess = True

            decay_value = decay[0]*(u_max-u_min)
            u_min_decay,u_max_decay = u_min+decay_value,u_max-decay_value

            decay_value = decay[1]*(v_max-v_min)
            v_min_decay,v_max_decay = v_min+decay_value,v_max-decay_value
            
            # -------------------------------
        
            uu = np.linspace(u_min, u_max, n_u)
            vv = np.linspace(v_min, v_max, n_v)
            U, V = np.meshgrid(uu, vv)

            Xs = radius * np.sin(V) * np.cos(U)
            Ys = radius * np.sin(V) * np.sin(U)
            Zs = radius * np.cos(V)
            grid_local = np.stack([Xs, Ys, Zs], axis=2)

            sampled_points = grid_local.reshape(-1,3) @ R.T    
            sampled_points = sampled_points.reshape(n_v, n_u, 3) + center

            # -------------------------------
   
            uu = np.linspace(u_min_decay, u_max_decay, n_u)
            vv = np.linspace(v_min_decay, v_max_decay, n_v)
            U, V = np.meshgrid(uu, vv)

            Xs = radius * np.sin(V) * np.cos(U)
            Ys = radius * np.sin(V) * np.sin(U)
            Zs = radius * np.cos(V)
            grid_local = np.stack([Xs, Ys, Zs], axis=2)

            sampled_points_decay = grid_local.reshape(-1,3) @ R.T    
            sampled_points_decay = sampled_points_decay.reshape(n_v, n_u, 3) + center

            err = evaluate_fit_error(pts,sampled_points)
            err_list.append([err,sampled_points,sampled_points_decay,sucess])

        

        except Exception as e:
            pass
            # print(traceback.print_exc(e))

    if len(err_list)>0:
        return_data = sorted(err_list,key=lambda x:x[0],reverse=False)[0]
        err,sampled_points,sampled_points_decay,sucess = return_data
        return err,sampled_points_decay,sucess
    else:
        return 999,None,False

def evaluate_fit_error(orig_points, fit_points, boundary_weight=3.0):

    if len(fit_points.shape)==3:
        H, W, _ = fit_points.shape
    else:
        H = W = fit_points.shape[0]**0.5

    orig_points = orig_points.reshape(-1,3)
    fit_points = fit_points.reshape(-1,3)

    min_coords = orig_points.min(axis=0)
    max_coords = orig_points.max(axis=0)
    bbox_size = max_coords - min_coords
    diag_len = np.linalg.norm(bbox_size)

    weight = np.ones((H, W))
    weight[0, :]   = boundary_weight      # top
    weight[-1, :]  = boundary_weight      # bottom
    weight[:, 0]   = boundary_weight      # left
    weight[:, -1]  = boundary_weight      # right
    weight = weight.reshape(-1)  

    tree_fit = cKDTree(fit_points)
    d1, _ = tree_fit.query(orig_points)
    tree_orig = cKDTree(orig_points)
    d2, _ = tree_orig.query(fit_points)

    mean_err = 0.5 * (np.sum(d1 * weight) + np.sum(d2 * weight)) / (np.sum(weight) * diag_len)
    # print(mean_err)
    return mean_err

###########
def rotate_points_around_axis(points, point_on_axis, axis_dir, angle):

    axis = axis_dir / np.linalg.norm(axis_dir)
    pts = points.reshape(-1,3)
    p_rel = pts - point_on_axis
    cos = np.cos(angle)
    sin = np.sin(angle)
    k_dot_v = np.dot(p_rel, axis)
    cross_k_v = np.cross(axis, p_rel)
    v_rot = p_rel * cos + cross_k_v * sin + np.outer(k_dot_v, axis) * (1 - cos)
    return (v_rot + point_on_axis).reshape(points.shape)

def finetune_plane_pair(points_a,points_b,type_a,type_b,degree = 0,ref_a_to_b_vec=np.array([0,0,0])):
    # input plane plane or plane,other

    # -------------------------

    pts_a = points_a.reshape(-1,3)
    center_a = pts_a.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_a - center_a, full_matrices=False)
    normal_a = Vt[-1]
    normal_a = normal_a / np.linalg.norm(normal_a)

    pts_b = points_b.reshape(-1,3)
    center_b = pts_b.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_b - center_b, full_matrices=False)
    normal_b = Vt[-1]
    normal_b = normal_b / np.linalg.norm(normal_b)

    # -------------------------

    if type_a=="plane" and type_b=="plane":
        cos_angle = np.clip(normal_a.dot(normal_b), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle > 90:
            normal_b = -normal_b
            cos_angle = -cos_angle   
            angle = 180 - angle
        if angle > 5: return points_a, points_b

    # -------------------------

    d2 = cdist(pts_a, pts_b, metric='sqeuclidean')
    k=8
    flat_idx = np.argpartition(d2.ravel(), k)[:k]
    ia, ib = np.unravel_index(flat_idx, d2.shape)
    pts_a_selected = pts_a[ia]
    pts_b_selected = pts_b[ib]
    vec_a = pts_a_selected - center_a
    vec_b = pts_b_selected - center_b
    vec_a = vec_a.mean(axis=0)
    vec_b = vec_b.mean(axis=0)
    vec_a /= np.linalg.norm(vec_a)
    vec_b /= np.linalg.norm(vec_b)

    # -------------------------

    axis_a = np.cross(vec_a, normal_a)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(vec_b, normal_b)
    axis_b /= np.linalg.norm(axis_b)

    if type_b!="plane" and np.linalg.norm(ref_a_to_b_vec) > 1e-8:
        rotated_vec = rotate_points_around_axis(vec_a[None,:], np.zeros(3), axis_a, degree)[0]
        if np.dot(rotated_vec, ref_a_to_b_vec) < np.dot(vec_a, ref_a_to_b_vec):
            axis_a = -axis_a

    if type_a!="plane" and np.linalg.norm(-ref_a_to_b_vec) > 1e-8:
        rotated_vec = rotate_points_around_axis(vec_b[None,:], np.zeros(3), axis_b, degree)[0]
        if np.dot(rotated_vec, -ref_a_to_b_vec) < np.dot(vec_b, -ref_a_to_b_vec):
            axis_b = -axis_b

    # -------------------------

    dis_a = np.linalg.norm(center_a-pts_a_selected.mean(axis=0))
    dis_b = np.linalg.norm(center_b-pts_b_selected.mean(axis=0))
    rot_center_a = center_a - vec_a * 0.9*dis_a
    rot_center_b = center_b - vec_b * 0.9*dis_b
    if type_a=="plane":
        points_a = rotate_points_around_axis(points_a, rot_center_a, axis_a, degree)
    if type_b=="plane":
        points_b = rotate_points_around_axis(points_b, rot_center_b, axis_b, degree)
    # print("sucess")
    return points_a,points_b


def fit_bspline_surface(
        surf,
        surf_mask,
        fit_tolerance_array,
        use_fit_plane,
        use_fit_cylinder,
        use_fit_cone,
        use_fit_torus,
        use_fit_sphere,
        use_plane_finetune,
        use_cylinder_decay,
        use_cone_decay,
        use_torus_decay,
        use_sphere_decay,
        plane_finetune_array,
        cylinder_decay_array,
        cone_decay_array,
        torus_decay_array,
        sphere_decay_array,
        num_uv = 16
    ):
    ###########################################
    fit_faces_params = []
    valid = sum(surf_mask)
    for i,points in enumerate(surf):
        DegMin,DegMax = 2,8
        fit_sucess = False
        face_type = "bspline"
        tolerance = fit_tolerance_array[i]
        cylinder_decay = cylinder_decay_array[i] if use_cylinder_decay else 0
        cone_decay = cone_decay_array[i] if use_cone_decay else 0
        torus_decay = torus_decay_array[i] if use_torus_decay else [0,0]
        sphere_decay = sphere_decay_array[i] if use_sphere_decay else [0,0]
        
        if i>=valid:continue
        fit_data={}
        if use_fit_plane: 
            plane_err,plane_points,plane_sucess = fit_plane(points)
            if plane_sucess: fit_data["plane"] = [plane_err,plane_points]
        if use_fit_cylinder:
            cylinder_err,cylinder_points,cylinder_sucess = fit_cylinder(points,decay=cylinder_decay)
            if cylinder_sucess: fit_data["cylinder"] = [cylinder_err,cylinder_points]
        if use_fit_cone:
            cone_err,cone_points,cone_sucess = fit_cone(points,decay=cone_decay)
            if cone_sucess: fit_data["cone"] = [cone_err,cone_points]
        if use_fit_torus:
            torus_err,torus_points,torus_sucess = fit_torus(points,decay=torus_decay)
            if torus_sucess: fit_data["torus"] = [torus_err,torus_points]
        if use_fit_sphere:
            sphere_err,sphere_points,sphere_sucess = fit_sphere(points,decay=sphere_decay)
            if sphere_sucess: fit_data["sphere"] = [sphere_err,sphere_points]

        best_type, (best_err, best_points) = min(fit_data.items(),key=lambda kv: kv[1][0])

        if use_fit_plane and best_type == "plane" and best_err<0.025:
            points,DegMin,tolerance,fit_sucess,face_type = best_points,0,1e-3,True,"plane"
            print(i,"plane")
        elif use_fit_cylinder and best_type == "cylinder" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"cylinder"
            print(i,"cylinder")
            print(i,cylinder_decay)
        elif use_fit_cylinder and best_type == "cone" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"cone"
            print(i,"cone")
            print(i,cone_decay)
        elif use_fit_cylinder and best_type == "torus" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"torus"
            print(i,"torus")
            print(i,torus_decay)
        elif use_fit_cylinder and best_type == "sphere" and best_err<0.025 :
            points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"sphere"
            print(i,"sphere")
            print(i,sphere_decay)

        if not fit_sucess:   
         
            points = laplacian_smooth_points(points)
            if use_fit_plane and best_type == "plane" and best_err<0.025:
                points,DegMin,tolerance,fit_sucess,face_type = best_points,0,1e-3,True,"plane"
                print(i,"plane")
            elif use_fit_cylinder and best_type == "cylinder" and best_err<0.025 :
                points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"cylinder"
                print(i,"cylinder")
                print(i,cylinder_decay)
            elif use_fit_cylinder and best_type == "cone" and best_err<0.025 :
                points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"cone"
                print(i,"cone")
                print(i,cone_decay)
            elif use_fit_cylinder and best_type == "torus" and best_err<0.025 :
                points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"torus"
                print(i,"torus")
                print(i,torus_decay)
            elif use_fit_cylinder and best_type == "sphere" and best_err<0.025 :
                points,DegMin,tolerance,fit_sucess,face_type = best_points,6,1e-3,True,"sphere"
                print(i,"sphere")
                print(i,sphere_decay)

        fit_faces_params.append([points,DegMin,DegMax,tolerance,face_type])

    ###########################################
    if use_plane_finetune:
        for i in range(len(surf)):
            for j in range(i+1,len(surf)):
                if i>=valid or j>=valid:continue
                if plane_finetune_array[i][j][0]>0:
                    if fit_faces_params[i][4] !="plane" or fit_faces_params[j][4]!="plane":continue  #TODO 先把平曲微调关掉
                    print("{},{}".format(i,j),plane_finetune_array[i][j])
                    fit_faces_params[i][0],fit_faces_params[j][0] = finetune_plane_pair(fit_faces_params[i][0],fit_faces_params[j][0],fit_faces_params[i][4],fit_faces_params[j][4],plane_finetune_array[i][j][0],plane_finetune_array[i][j][1:])

    ###########################################
    recon_faces = []
    face_types = []
    for fit_face_params in fit_faces_params:
        points,DegMin,DegMax,tolerance,face_type = fit_face_params
        face_types.append(face_type)
        uv_points_array = TColgp_Array2OfPnt(1, num_uv, 1, num_uv)
        for u_index in range(1, num_uv+1):
            for v_index in range(1, num_uv+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, DegMin, DegMax, GeomAbs_C2, tolerance).Surface()
        face = BRepBuilderAPI_MakeFace(approx_face,1e-8).Face()
        recon_faces.append(face)

    return recon_faces, face_types

def get_adj_faces_distance(faces,adj_matrix):
    max_distance={i:0 for i in range(len(faces))}
    for i in range(len(faces)):
        for j in range(len(faces)):
            if adj_matrix[i][j]==1:
                dis = BRepExtrema_DistShapeShape(faces[i], faces[j]).Value()
                if dis>max_distance[i]: max_distance[i] = dis
    return max_distance

# def extend_faces_and_make_params(faces,max_distance,extend_length_dis,extend_length_const):  
#     face_map_params={}
#     for i in range(len(faces)):
#         modified_face=TopoDS_Face()
#         extend_length = max_distance[i]*extend_length_dis + extend_length_const   
#         breplib_ExtendFace(faces[i], extend_length, True, True, True, True, modified_face)
#         face_map_params.update({i:(faces[i],modified_face)})
#     return face_map_params

def extend_faces_and_make_params(faces,max_distance,extend_length_dis,extend_length_const):  
    face_map_params={}
    for i in range(len(faces)):
        if i==6:
            pass
        u_len, v_len = get_face_uv_len(faces[i])
        modified_face=TopoDS_Face()
        extend_length = max_distance[i]*extend_length_dis + extend_length_const*(0.8+u_len/3)   
        breplib_ExtendFace(faces[i], extend_length, True, True, False, False, modified_face)
        extend_length = max_distance[i]*extend_length_dis + extend_length_const*(0.8+v_len/3)  
        breplib_ExtendFace(modified_face, extend_length, False, False, True, True, modified_face)
        face_map_params.update({i:(faces[i],modified_face)})
    return face_map_params

############################################

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
                if np.any(np.array(v_dis)<3e-2):
                    continue
                else:
                    del_flag = True
                    break
            if del_flag:
                print("del1")
                return_results[i][0].remove(face)

    return return_results


def copy_list_list(data_list):
    new_list = [[],[],[],[]]
    for i,data in enumerate(data_list):
        for d in data:
            new_list[i].append(d)
    return new_list


def del_surf_from_edge_degree_3(results): 
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
                        # return []
                        #
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
                                    
                                    if len(set(edge_to_face[e]) - set(temp_del_data[0]))+1>=3: 
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
                    print("not del when appear 3edge")

            if get_del_face:
                break            
        else:
            end_flag = True
            break
    #
    for f in del_data[0]:
        main_index = face_to_main_idx[f]
        results[main_index][0].remove(f)
        print("del3")
    return results
                                
def final_fuse(faces,orign_face):#
    if len(faces)==1:
        return faces[0]
    edge_select=[]
    for f in faces:  
        edges = get_edges_from_face(f)
        for e in edges:
            for e_in in edge_select:
                if are_edges_same(e,e_in,tolerance=1e-4):
                    edge_select.remove(e_in)
                    print("bool cut edge")
                    break
                if is_edge_in(e,e_in,tolerance=1e-4):
                    edge_select.remove(e_in)
                    new_e = fundamental_bop(e_in,e,op_name="cut")
                    edge_select.append(new_e)
                    print("bool cut edge")
                    break
                if is_edge_in(e_in,e,tolerance=1e-4):
                    edge_select.remove(e_in)
                    new_e = fundamental_bop(e,e_in,op_name="cut")
                    edge_select.append(new_e)
                    print("bool cut edge")
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

##############################################################
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
            if are_edges_same(ei,ej,tolerance=5e-2) or is_edge_in(ei,ej,tolerance=5e-2) or is_edge_in(ej,ei,tolerance=5e-2):
                break
        else:
            # display.DisplayShape(all_group_edges[i],color=0,transparency=False,update=True)
            return False
    return True   

def check_brep_topo_validity(results):
    for result in results:
        if len(result[0])==0:
            return False
    return True

def check_brep_validity(step_file_path):
    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True
    brep_closed_ok = True  # Initialize closed BRep check
    solid_one_ok = True

    # 1. Check if BRep has more than one solid
    if isinstance(step_file_path, str):
        # Read the STEP file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file_path)
        
        if status != IFSelect_RetDone:
            print("Error: Unable to read STEP file")
            return False
        
        step_reader.TransferRoot()
        num_shapes = step_reader.NbShapes()
        if num_shapes !=1:
            return False
        shape = step_reader.Shape()

    elif isinstance(step_file_path, TopoDS_Solid):
        shape = step_file_path

    else:
        return False

    # 2. Check all wires
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())

            # Create a ShapeFix_Wire object
            wire_fixer = ShapeFix_Wire(wire, face, 0.01)
            wire_fixer.Load(wire)
            wire_fixer.SetFace(face)
            wire_fixer.SetPrecision(0.01)
            wire_fixer.SetMaxTolerance(1)
            wire_fixer.SetMinTolerance(0.0001)

            # Fix the wire
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()

            # Analyze the fixed wire
            wire_analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            wire_analysis.Load(fixed_wire)
            wire_analysis.SetPrecision(0.01)
            wire_analysis.SetSurface(BRep_Tool.Surface(face))

            # 1. Check wire edge order
            order_status = wire_analysis.CheckOrder()
            if order_status != 0:  # 0 means no error
                # print(f"Wire order issue detected: {order_status}")
                wire_order_ok = False

            # 2. Check wire self-intersection
            if wire_analysis.CheckSelfIntersection():
                wire_self_intersection_ok = False

            wire_explorer.Next()
        face_explorer.Next()

    # 3. Check for bad edges in shells
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = topods_Shell(shell_explorer.Current())
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(shell)

        if shell_analysis.HasBadEdges():
            shell_bad_edges_ok = False

        shell_explorer.Next()

    # 4. Check if BRep is closed (no free edges)
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = free_bounds.GetOpenWires()
    edge_explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    num_free_edges = 0
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        num_free_edges += 1
        # print(f"Free edge: {edge}")
        edge_explorer.Next()
    if num_free_edges > 0:
        brep_closed_ok = False

    return int(wire_order_ok and wire_self_intersection_ok and shell_bad_edges_ok and brep_closed_ok and solid_one_ok)

################################################################
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

SURFACE_TYPES = ["plane", "cylinder", "cone", "torus", "sphere", "bspline"]

def empty_type_stats():
    return {surface_type: {"success": 0, "total": 0} for surface_type in SURFACE_TYPES}

def empty_pair_type_stats():
    stats = {}
    for i, type_a in enumerate(SURFACE_TYPES):
        for type_b in SURFACE_TYPES[i:]:
            stats["{}-{}".format(type_a, type_b)] = {"success": 0, "total": 0}
    return stats

def pair_type_key(type_a, type_b):
    type_a = type_a if type_a in SURFACE_TYPES else "bspline"
    type_b = type_b if type_b in SURFACE_TYPES else "bspline"
    idx_a = SURFACE_TYPES.index(type_a)
    idx_b = SURFACE_TYPES.index(type_b)
    if idx_a <= idx_b:
        return "{}-{}".format(type_a, type_b)
    return "{}-{}".format(type_b, type_a)

def merge_nested_stats(dst, src):
    for key, value in src.items():
        if key not in dst:
            dst[key] = {"success": 0, "total": 0}
        dst[key]["success"] += value.get("success", 0)
        dst[key]["total"] += value.get("total", 0)

def add_rates(stats):
    return {
        key: {
            "success": value["success"],
            "total": value["total"],
            "rate": value["success"] / value["total"] if value["total"] else 0.0,
        }
        for key, value in stats.items()
    }

def get_adjacent_pairs(adj_matrix, valid_count=None):
    if valid_count is None:
        valid_count = len(adj_matrix)
    pairs = set()
    for i in range(valid_count):
        for j in range(i + 1, valid_count):
            if adj_matrix[i][j] == 1 or adj_matrix[j][i] == 1:
                pairs.add((i, j))
    return pairs

def get_intersection_type_stats(adjacent_pairs, failed_intersection_pairs, face_types):
    stats = empty_pair_type_stats()
    for i, j in adjacent_pairs:
        key = pair_type_key(face_types[i], face_types[j])
        stats[key]["total"] += 1
        if (i, j) not in failed_intersection_pairs:
            stats[key]["success"] += 1
    return stats

def get_intersection_surface_type_stats(adjacent_pairs, failed_intersection_pairs, face_types):
    stats = empty_type_stats()
    for i, j in adjacent_pairs:
        success = (i, j) not in failed_intersection_pairs
        for idx in (i, j):
            face_type = face_types[idx] if face_types[idx] in SURFACE_TYPES else "bspline"
            stats[face_type]["total"] += 1
            if success:
                stats[face_type]["success"] += 1
    return stats

def get_loop_type_stats(face_types, success_indices):
    # Face-level loop stats: one main face contributes one total count, and
    # contributes one success count only when its cut result is not error.
    stats = empty_type_stats()
    for idx, face_type in enumerate(face_types):
        face_type = face_type if face_type in SURFACE_TYPES else "bspline"
        stats[face_type]["total"] += 1
        if idx in success_indices:
            stats[face_type]["success"] += 1
    return stats

def nested_success_total(stats):
    return sum(value["success"] for value in stats.values())

def make_solid_stats(index, total_faces, total_intersection_pairs=0):
    return {
        "index": int(index),
        "total_faces": int(total_faces),
        "total_intersection_pairs": int(total_intersection_pairs),
        "intersection_success_pairs": 0,
        "intersection_by_surface_type": empty_type_stats(),
        "intersection_by_pair_type": empty_pair_type_stats(),
        "loop_success_faces": 0,
        "loop_by_surface_type": empty_type_stats(),
        "solid_loop_success": 0,
        "attempts": 0,
        "timeout": 0,
        "exception": "",
    }

def make_empty_solid_stats(index, surf_mask, adj_matrix):
    valid_faces = int(sum(surf_mask))
    return make_solid_stats(index, valid_faces, len(get_adjacent_pairs(adj_matrix, valid_faces)))

def update_stats_from_result(total, item):
    total["solids"] += 1
    total["faces"] += item["total_faces"]
    total["intersection_pairs"] += item["total_intersection_pairs"]
    total["intersection_success_pairs"] += item["intersection_success_pairs"]
    total["loop_success_faces"] += item["loop_success_faces"]
    total["solid_loop_success"] += item["solid_loop_success"]
    total["timeouts"] += item["timeout"]
    merge_nested_stats(total["intersection_by_surface_type"], item["intersection_by_surface_type"])
    merge_nested_stats(total["intersection_by_pair_type"], item["intersection_by_pair_type"])
    merge_nested_stats(total["loop_by_surface_type"], item["loop_by_surface_type"])

def rate_text(num, den):
    if den == 0:
        return "0/0=0.00%"
    return "{}/{}={:.2f}%".format(num, den, num / den * 100.0)

def stats_postfix(total):
    return {
        "intersect_pair": rate_text(total["intersection_success_pairs"], total["intersection_pairs"]),
        "loop_face": rate_text(total["loop_success_faces"], total["faces"]),
        "timeout": total["timeouts"],
    }

def build_summary(start, end, solid_workers, face_workers, total_stats):
    return {
        "start": int(start),
        "end": int(end),
        "solid_workers": int(solid_workers),
        "face_workers": int(face_workers),
        "processed_solids": int(total_stats["solids"]),
        "total": total_stats,
        "intersection_pair_success_rate": (
            total_stats["intersection_success_pairs"] / total_stats["intersection_pairs"]
            if total_stats["intersection_pairs"] else 0.0
        ),
        "loop_face_success_rate": (
            total_stats["loop_success_faces"] / total_stats["faces"]
            if total_stats["faces"] else 0.0
        ),
        "intersection_success_rate": (
            total_stats["intersection_success_pairs"] / total_stats["intersection_pairs"]
            if total_stats["intersection_pairs"] else 0.0
        ),
        "loop_success_rate": (
            total_stats["loop_success_faces"] / total_stats["faces"]
            if total_stats["faces"] else 0.0
        ),
        "solid_loop_success_rate": (
            total_stats["solid_loop_success"] / total_stats["solids"]
            if total_stats["solids"] else 0.0
        ),
        "intersection_by_surface_type": add_rates(total_stats["intersection_by_surface_type"]),
        "intersection_by_pair_type": add_rates(total_stats["intersection_by_pair_type"]),
        "loop_by_surface_type": add_rates(total_stats["loop_by_surface_type"]),
    }

def write_stats_snapshot(stats_path, start, end, solid_workers, face_workers, total_stats):
    summary = build_summary(start, end, solid_workers, face_workers, total_stats)
    tmp_path = stats_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, stats_path)
    return summary

def process_one_face(arguments):
    now_face, cut_faces,use_ef_section = arguments
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
 
    result=cut_face_and_select(now_face,cut_faces,use_ef_section)
    stop_event.set()  

    return result

def process_one_solid(
        surf,
        surf_mask,
        adj_matrix,
        index,
        save_num,
        save_dir,
        face_workers,
        use_fit_plane,
        use_fit_cylinder,
        use_fit_cone,
        use_fit_torus,
        use_fit_sphere
    ):
    valid_faces = int(sum(surf_mask))
    adjacent_pairs = get_adjacent_pairs(adj_matrix, valid_faces)
    stats = make_solid_stats(index, valid_faces, len(adjacent_pairs))
    selected_intersection_success_count = 0
    selected_face_success_count = 0
    selected_intersection_surface_type_stats = empty_type_stats()
    selected_intersection_type_stats = empty_pair_type_stats()
    selected_loop_type_stats = empty_type_stats()


    cut_params=[
        [1.0,0.08,5e-2,0],  
        [1.2,0.13,5e-2,0],
        [1.5,0.15,5e-2,0],
        # [1.5,0.20,5e-2,0],
        # [0.5,0.05,5e-2,1],  
        # [0.5,0.06,5e-2,0],
        # [0.6,0.08,5e-2,0],
        # [0.8,0.09,5e-2,0],
        # [1.0,0.10,5e-2,2],
        # [0.5,0.06,5e-2,2],
        ]

    sew_params=[
        [1e-5],
        [1e-3],
    ]

    mode = 3 
    use_display = False
    if use_display: 
        display, start_display, add_menu, add_function_to_menu = init_display()

    use_plane_finetune = False
    plane_finetune_once = 0.05
    use_cylinder_decay = True
    cylinder_decay_once = 0.05
    use_cone_decay = True
    cone_decay_once = 0.05
    use_torus_decay = True
    torus_decay_once = [0.05,0.05]
    use_sphere_decay = True
    sphere_decay_once = [0.05,0.05]

    use_final_fuse = False  

    #################################################### 
    
    all_results = [] 
    fit_tolerance_array=np.ones((len(surf),)) *cut_params[0][2] 
    use_ef_section_array=np.zeros((len(surf),),dtype=np.bool_)  
    plane_finetune_array=np.zeros((len(surf),len(surf),4),dtype=np.float64) 
    cylinder_decay_array = np.zeros((len(surf),),dtype=np.float32) 
    cone_decay_array = np.zeros((len(surf),),dtype=np.float32)
    torus_decay_array = np.zeros((len(surf),2),dtype=np.float32)
    sphere_decay_array = np.zeros((len(surf),2),dtype=np.float32)
    for now_num in range(len(cut_params)):
        stats["attempts"] += 1
        try :
            extend_length_dis = cut_params[now_num][0]
            extend_length_const = cut_params[now_num][1]
            spline_fit_tolerance = cut_params[now_num][2]
            reset_flag = cut_params[now_num][3]
            if reset_flag==1:  
                plane_finetune_array=np.zeros((len(surf),len(surf),4),dtype=np.float64)
                cylinder_decay_array = np.zeros((len(surf),),dtype=np.float32)
                cone_decay_array = np.zeros((len(surf),),dtype=np.float32)
                torus_decay_array = np.zeros((len(surf),2),dtype=np.float32)
                sphere_decay_array = np.zeros((len(surf),2),dtype=np.float32)
            elif reset_flag==2:
                plane_finetune_array[:,:,0] = 0.15
                plane_finetune_array[:,:,1:] = np.array([0,0,0])
                cylinder_decay_array[:] = 0.15
                cone_decay_array[:] = 0.15
                torus_decay_array[:,0],torus_decay_array[:,1] = 0.15,0.15
                sphere_decay_array[:,0],sphere_decay_array[:,1] = 0.15,0.15

            recon_faces, face_types = fit_bspline_surface(
                surf,
                surf_mask,
                fit_tolerance_array,
                use_fit_plane,
                use_fit_cylinder,
                use_fit_cone,
                use_fit_torus,
                use_fit_sphere,
                use_plane_finetune,
                use_cylinder_decay,
                use_cone_decay,
                use_torus_decay,
                use_sphere_decay,
                plane_finetune_array,
                cylinder_decay_array,
                cone_decay_array,
                torus_decay_array,
                sphere_decay_array,
            )
            max_distance = get_adj_faces_distance(recon_faces,adj_matrix)
            all_face_params = extend_faces_and_make_params(recon_faces, max_distance, extend_length_dis, extend_length_const)

            
            face_and_cut_faces = []
            for i in range(len(recon_faces)):
                connect_faces_params_list = []
                for j in range(len(recon_faces)):
                    if i!=j and adj_matrix[i][j]==1:
                        connect_faces_params_list.append({j:all_face_params[j]})
                main_face_params = {i:all_face_params[i]}
                use_ef_section = use_ef_section_array[i]
                face_and_cut_faces.append([main_face_params,connect_faces_params_list,use_ef_section])

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
                                print(f"overtime")
                                future.cancel()  
                                executor.shutdown(wait=False,cancel_futures=True)  
                                is_error = True
                                break
                            except Exception as e:
                                future.cancel() 
                                is_error = True
                                print(f"error")
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
                failed_intersection_pairs = set()
                face_success_indices = set()

               
                all_main_index = [i for i in range(len(surf))]
                decay_change_over = []
                finetune_change_over = []
                for res_faces, error_flag, main_index, face_section_error_cut_index, edge_section_error_cut_index in results:  
                    all_main_index.remove(main_index)
                    error_sum += int(error_flag)
                    for cut_index in face_section_error_cut_index:
                        failed_intersection_pairs.add(tuple(sorted((main_index, cut_index))))
                    if not error_flag:
                        face_success_indices.add(main_index)
                    if error_flag==True :
                 
                        if now_num<len(cut_params)-1: 
                            fit_tolerance_array[main_index] = cut_params[now_num+1][2] 
                            for idx in face_section_error_cut_index:
                                fit_tolerance_array[idx] = cut_params[now_num+1][2]

                                                    
                        use_ef_section_array[main_index] = True 

                   
                        for idx in face_section_error_cut_index:
                            if idx not in finetune_change_over and main_index not in finetune_change_over:
                                plane_finetune_array[main_index,idx][0] = min(plane_finetune_array[main_index,idx][0] + plane_finetune_once,0.15) 
                                plane_finetune_array[idx,main_index][0] = min(plane_finetune_array[idx,main_index][0] + plane_finetune_once,0.15) 
                                finetune_change_over.append(idx)
                                finetune_change_over.append(main_index)
                        for idx_pair in edge_section_error_cut_index:
                            if idx_pair[0] not in finetune_change_over and idx_pair[1] not in finetune_change_over:
                                plane_finetune_array[idx_pair[0],idx_pair[1]][0] = min(plane_finetune_array[idx_pair[0],idx_pair[1]][0]  + plane_finetune_once,0.15) 
                                plane_finetune_array[idx_pair[1],idx_pair[0]][0] = min(plane_finetune_array[idx_pair[1],idx_pair[0]][0]  + plane_finetune_once,0.15) 
                                plane_finetune_array[idx_pair[0],idx_pair[1]][1:] = idx_pair[2]  #从i到j的向量
                                plane_finetune_array[idx_pair[1],idx_pair[0]][1:] = -idx_pair[2]
                                finetune_change_over.append(idx_pair[0])
                                finetune_change_over.append(idx_pair[1])
                     
                        if main_index not in decay_change_over:
                            cylinder_decay_array[main_index]= np.clip(cylinder_decay_array[main_index] + cylinder_decay_once,None,0.15)
                            cone_decay_array[main_index]= np.clip(cone_decay_array[main_index] + cone_decay_once,None,0.15)
                            torus_decay_array[main_index]= np.clip(torus_decay_array[main_index] + torus_decay_once,None,[0.15,0.15])
                            sphere_decay_array[main_index]= np.clip(sphere_decay_array[main_index] + sphere_decay_once,None,[0.15,0.15])

                        connect_faces_params_list = face_and_cut_faces[main_index][1]
                        decay_change_over.append(main_index)
                        for cut_face_params in connect_faces_params_list:
                            j = list(cut_face_params.keys())[0]
                            if j not in decay_change_over:
                                cylinder_decay_array[j]= np.clip(cylinder_decay_array[j] + cylinder_decay_once,None,0.15)
                                cone_decay_array[j]= np.clip(cone_decay_array[j] + cone_decay_once,None,0.15)
                                torus_decay_array[j]= np.clip(torus_decay_array[j] + torus_decay_once,None,[0.15,0.15])
                                sphere_decay_array[j]= np.clip(sphere_decay_array[j] + sphere_decay_once,None,[0.15,0.15])
                                decay_change_over.append(j)
                
       
                current_intersection_type_stats = get_intersection_type_stats(adjacent_pairs, failed_intersection_pairs, face_types)
                current_intersection_surface_type_stats = get_intersection_surface_type_stats(adjacent_pairs, failed_intersection_pairs, face_types)
                current_loop_type_stats = get_loop_type_stats(face_types, face_success_indices)
                intersection_success_count = len(adjacent_pairs - failed_intersection_pairs)
                selected_intersection_success_count = intersection_success_count
                selected_intersection_surface_type_stats = current_intersection_surface_type_stats
                selected_intersection_type_stats = current_intersection_type_stats
                selected_face_success_count = len(face_success_indices)
                selected_loop_type_stats = current_loop_type_stats

                for other_main_index in all_main_index:
                    if now_num<len(cut_params)-1:
                        fit_tolerance_array[other_main_index] = cut_params[now_num+1][2]
                    if other_main_index not in decay_change_over:
                        cylinder_decay_array[other_main_index]= np.clip(cylinder_decay_array[other_main_index] + cylinder_decay_once,None,0.15)
                        cone_decay_array[other_main_index]= np.clip(cone_decay_array[other_main_index] + cone_decay_once,None,0.15)
                        torus_decay_array[other_main_index]= np.clip(torus_decay_array[other_main_index] + torus_decay_once,None,[0.15,0.15])
                        sphere_decay_array[other_main_index]= np.clip(sphere_decay_array[other_main_index] + sphere_decay_once,None,[0.15,0.15])
                        decay_change_over.append(other_main_index)
                    
        
                if error_sum==0 and not is_error:
                    results = sorted(results,key=lambda x:x[2],reverse=False) 
                    results = del_surf_from_edge_degree_3(results)
                    results = del_surf_from_vert_degree_1(results,adj_matrix)

                
                    topo_valid = check_brep_topo_validity(results)
                    if not topo_valid:
                        print("topo error")

                    edge_valid = check_brep_validity_by_edge(results)
                    if not edge_valid:
                        print("edge degree 1 error")
                                    
            
                    all_res_faces = []
                    for result in results:
                        all_res_faces.extend(result[0])

                    if edge_valid and topo_valid:
                  
                        if use_final_fuse:
                            all_res_faces = []
                            for result in results:
                                new_face = final_fuse(result[0],all_face_params[main_index][1])
                                all_res_faces.append(new_face)
                        
                   
                        solid = None
                        for sew_num in range(len(sew_params)):
                            try:
                                sewing_tolerance = sew_params[sew_num][0]
                                solid = sewing_cutted_faces(all_res_faces,sewing_tolerance)
                                break
                            except:
                                print("error")
                    
                        if isinstance(solid,TopoDS_Shell) or isinstance(solid,TopoDS_Solid):
                            write_step_file(solid, '{}/{}_{}_{}.step'.format(save_dir,len(results),save_num,index))
                            write_stl_file(solid, '{}/{}_{}_{}.stl'.format(save_dir,len(results),save_num,index), linear_deflection=0.001, angular_deflection=0.5)
                            success = True
                        else:
                            print("unclosed," + ":{}".format(now_num))
                    else:
                        for index_all in range(len(surf)):
                            if index_all not in decay_change_over:
                                cylinder_decay_array[index_all]= min(cylinder_decay_array[index_all] + cylinder_decay_once,0.15)

                else:
                    print(":{}".format(now_num))
                all_results.append([error_sum,results])
                if success:
                    stats["solid_loop_success"] = 1
                    break

        except Exception as e:
            if now_num<len(cut_params)-1:
                fit_tolerance_array=np.ones((len(surf),)) *cut_params[now_num+1][2]
            cylinder_decay_array = cylinder_decay_once
            cylinder_decay_array = min(cylinder_decay_once*(now_num+1),0.15)
            print(":", type(e).__name__)
            print(":", e)
            print(traceback.print_exc(e))
            stats["exception"] = "{}: {}".format(type(e).__name__, e)
            
    if use_display:
        if mode==3 and len(all_results)>0: 
            all_results= sorted(all_results,key=lambda x:x[0],reverse=False)
            error_num, results = all_results[0]
            if error_num!=0: 
                results = sorted(results,key=lambda x:x[2],reverse=False) 
                all_res_faces = []
                for result in results:
                    all_res_faces.extend(result[0])
            
            display.EraseAll()
            for face in all_res_faces:
                display.DisplayShape(face, update=True,color=0,transparency=0.5)
            display.FitAll()

        start_display()

    stats["intersection_success_pairs"] = int(selected_intersection_success_count)
    stats["intersection_by_surface_type"] = selected_intersection_surface_type_stats
    stats["intersection_by_pair_type"] = selected_intersection_type_stats
    stats["loop_success_faces"] = int(selected_face_success_count)
    stats["loop_by_surface_type"] = selected_loop_type_stats
    return stats

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

    surfs = surfs[start:end]
    surf_masks = surf_masks[start:end]
    adj_matrices = adj_matrices[start:end]
    indice = np.arange(len(surfs)) + start
    save_name = [args.name]*len(surfs)
    save_dir = [args.exp_path]*len(surfs)
    os.makedirs(args.exp_path,exist_ok=True)

    face_workers = args.face_workers
    solid_workers = args.solid_workers
    use_fit_plane = args.use_fit_plane
    use_fit_cylinder = args.use_fit_cylinder
    use_fit_cone = args.use_fit_cone
    use_fit_torus = args.use_fit_torus
    use_fit_sphere = args.use_fit_sphere

    #排个序
    mask_sizes = surf_masks.sum(axis=1)
    order = np.argsort(mask_sizes)
    surfs = [surfs[i] for i in order]
    surf_masks = [surf_masks[i] for i in order]
    adj_matrices = [adj_matrices[i] for i in order]
    indice = indice[order]

    total_stats = {
        "solids": 0,
        "faces": 0,
        "intersection_pairs": 0,
        "intersection_success_pairs": 0,
        "intersection_by_surface_type": empty_type_stats(),
        "intersection_by_pair_type": empty_pair_type_stats(),
        "loop_success_faces": 0,
        "loop_by_surface_type": empty_type_stats(),
        "solid_loop_success": 0,
        "timeouts": 0,
    }
    stats_path = args.stats_path
    if stats_path is None:
        stats_path = os.path.abspath("cut_faces_v4_stats.json")

    if solid_workers != 1:
        try:
            with ProcessPoolExecutor(max_workers=solid_workers, initializer=initializer) as executor:
                futures = {
                        executor.submit(process_one_solid,
                                        surfs[i],
                                        surf_masks[i],
                                        adj_matrices[i],
                                        indice[i],
                                        save_name[i],
                                        save_dir[i],
                                        face_workers,
                                        use_fit_plane,
                                        use_fit_cylinder,
                                        use_fit_cone,
                                        use_fit_torus,
                                        use_fit_sphere
                                        ): i for i in range(len(surfs))
                }
                with tqdm(total=len(futures), desc="cut_faces_v4_stats") as pbar:
                    for future in as_completed(futures):
                        i = futures[future]
                        try:
                            result = future.result()
                        except TimeoutError:
                            result = make_empty_solid_stats(indice[i], surf_masks[i], adj_matrices[i])
                            result["timeout"] = 1
                        except Exception as e:
                            result = make_empty_solid_stats(indice[i], surf_masks[i], adj_matrices[i])
                            result["exception"] = "{}: {}".format(type(e).__name__, e)
                            traceback.print_exc()

                        update_stats_from_result(total_stats, result)
                        write_stats_snapshot(stats_path, start, end, solid_workers, face_workers, total_stats)
                        pbar.set_postfix(stats_postfix(total_stats))
                        pbar.update(1)

        except Exception as e:
            print("error type:", type(e).__name__)
            print("error info:", e)
            traceback.print_exc()
        gc.collect()
    else:
        with tqdm(total=len(surfs), desc="cut_faces_v4_stats") as pbar:
            for i in range(0, len(surfs)):
                try:
                    result = process_one_solid(
                        surfs[i],
                        surf_masks[i],
                        adj_matrices[i],
                        indice[i],
                        args.name,
                        args.exp_path,
                        face_workers,
                        use_fit_plane,
                        use_fit_cylinder,
                        use_fit_cone,
                        use_fit_torus,
                        use_fit_sphere
                    )
                except Exception as e:
                    result = make_empty_solid_stats(indice[i], surf_masks[i], adj_matrices[i])
                    result["exception"] = "{}: {}".format(type(e).__name__, e)
                    traceback.print_exc()

                update_stats_from_result(total_stats, result)
                write_stats_snapshot(stats_path, start, end, solid_workers, face_workers, total_stats)
                pbar.set_postfix(stats_postfix(total_stats))
                pbar.update(1)

    write_stats_snapshot(stats_path, start, end, solid_workers, face_workers, total_stats)

    print("intersection_pair_success_rate:", rate_text(total_stats["intersection_success_pairs"], total_stats["intersection_pairs"]))
    print("loop_face_success_rate:", rate_text(total_stats["loop_success_faces"], total_stats["faces"]))
    print("solid_loop_success_rate:", rate_text(total_stats["solid_loop_success"], total_stats["solids"]))
    print("intersection_by_surface_type:")
    for key, value in sorted(total_stats["intersection_by_surface_type"].items()):
        print("  {}: {}".format(key, rate_text(value["success"], value["total"])))
    print("intersection_by_pair_type:")
    for key, value in sorted(total_stats["intersection_by_pair_type"].items()):
        print("  {}: {}".format(key, rate_text(value["success"], value["total"])))
    print("loop_by_surface_type:")
    for key, value in sorted(total_stats["loop_by_surface_type"].items()):
        print("  {}: {}".format(key, rate_text(value["success"], value["total"])))
    print("stats_path:", stats_path)
    return

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
                                        face_workers,
                                        use_fit_plane,
                                        use_fit_cylinder,
                                        use_fit_cone,
                                        use_fit_torus,
                                        use_fit_sphere
                                        ): i for i in range(len(surfs))
                }
                for future in tqdm(as_completed(futures), total=len(futures), disable=True):
                    try:
                        result = future.result(timeout=180)
                    except TimeoutError:
                        print("进程超时")
                        pass

        except Exception as e:
            print("错误类型:", type(e).__name__)
            print("错误信息:", e)
            traceback.print_exc()
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
                face_workers,
                use_fit_plane,
                use_fit_cylinder,
                use_fit_cone,
                use_fit_torus,
                use_fit_sphere
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./brepgen_post_process/npycom")
    parser.add_argument("--exp_path", type=str, default="./brepgen_post_process/exp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2048)
    parser.add_argument("--name", type=int, default=0)
    parser.add_argument("--face_workers", type=int, default=8)
    parser.add_argument("--solid_workers", type=int, default=32)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--use_fit_plane", type=bool, default=True)
    parser.add_argument("--use_fit_cylinder", type=bool, default=True)
    parser.add_argument("--use_fit_cone", type=bool, default=True)
    parser.add_argument("--use_fit_torus", type=bool, default=True)
    parser.add_argument("--use_fit_sphere", type=bool, default=True)
    
    args = parser.parse_args()
    
    main(args)
    # python ./cut_faces_new2.py --data_path ./gen_res --exp_path ./exp --start 0 --end 1 --name 0
    # python ./brepgen_post_process/cut_faces_debug_new_new_new.py --data_path ./gen_res --exp_path ./exp --start 0 --end 1 --name 0
    
    
    
