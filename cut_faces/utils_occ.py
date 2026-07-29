import os
import numpy as np
from collections import defaultdict

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeSolid,BRepBuilderAPI_Sewing
from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Compound,topods_Face,topods_Edge,topods_Vertex
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common,BRepAlgoAPI_Section
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.BOPAlgo import BOPAlgo_Builder
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

################################################## io
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

################################################## judge
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

################################################## get
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
          
        #开始记录边和点
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

def get_vertice_from_edge(edge):
    vertice=[]
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods_Vertex(vertex_explorer.Current())  
        vertex_explorer.Next()
        vertice.append(vertex)
    return vertice         
       
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

############################## compute  
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

# def cut_faces(face_ori, cut_faces):
#     bop = BOPAlgo_BOP()
#     object_list = TopTools_ListOfShape()
#     object_list.Append(face_ori)
#     tool_list = TopTools_ListOfShape()
#     for face in cut_faces:
#         tool_list.Append(face)
    
#     bop.SetArguments(object_list)
#     bop.SetTools(tool_list)
#     bop.SetOperation(BOPAlgo_Operation.BOPAlgo_CUT) 
#     bop.SetFuzzyValue(1e-6)
#     bop.SetRunParallel(True)
#     bop.Perform()
#     result_shape = bop.Shape()

#     explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
#     if explorer.More():
#         return topods_Face(explorer.Current())
#     else:
#         return None

def cut_faces(face_ori,cut_faces,return_compound = False): #return topods face
    new_shape = face_ori
    for face in cut_faces:
        new_shape = BRepAlgoAPI_Cut(new_shape, face).Shape()
    if return_compound:
        return new_shape
    else:
        explorer = TopExp_Explorer(new_shape, TopAbs_FACE)
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


#################################### fix
def fix_face(face): #maybe useless
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
        # print(["result compound"])
        return sewn_shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()
    # print([type(solid)])
    return solid


