import numpy as np
import warnings
import traceback
from collections import defaultdict

from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Face,topods_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import  BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface)

from occwl.uvgrid import ugrid, uvgrid
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.face import Face
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper
from occwl.edge import Edge


def get_bbox(point_cloud):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
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
    return min_point, max_point


def build_face_adjacency(ef_adj):
    max_F = ef_adj.max().item() + 1
    adj = np.zeros((max_F, max_F), dtype=np.bool_)
    f1, f2 = ef_adj[:, 0], ef_adj[:, 1]  # (LE,)
    adj[f1, f2] = True
    adj[f2, f1] = True 
    return adj #LE,LE

def build_fef_adjacency(ef_adj):
    max_F = ef_adj.max().item() + 1
    adj = np.zeros((max_F, max_F), dtype=np.int32)

    if ef_adj.size == 0:
        return adj

    f1 = ef_adj[:, 0]
    f2 = ef_adj[:, 1]

    np.add.at(adj, (f1, f2), 1)
    np.add.at(adj, (f2, f1), 1)

    return adj

def get_face_type(face):
    surf_type = BRepAdaptor_Surface(face.topods_shape()).GetType()
    if surf_type == GeomAbs_Plane:
        return 0
    if surf_type == GeomAbs_Cylinder:
        return 1
    if surf_type == GeomAbs_Cone:
        return 2
    if surf_type == GeomAbs_Torus:
        return 3
    if surf_type == GeomAbs_Sphere:
        return 4
    if surf_type == GeomAbs_BSplineSurface:
        return 5
    return 6



def update_mapping(data_dict):
    """
    Remove unused key index from data dictionary.
    """
    dict_new = {}
    mapping = {}
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx)) - set(data_dict.keys()))))
    # if np.any(skipped_indices)!=0: print("skip")
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new
    return dict_new, mapping


def face_edge_adj(shape):
    """
    *** COPY AND MODIFIED FROM THE ORIGINAL OCCWL SOURCE CODE ***
    Extract face/edge geometry and create a face-edge adjacency
    graph from the given shape (Solid or Compound)

    YBC altered:
    extend face to have a ring-like outer boundary, save the original surface and face into face_dict

    Args:
    - shape (Shell, Solid, or Compound): Shape

    Returns:
    - face_dict: Dictionary of occwl faces, with face ID as the key
    - edge_dict: Dictionary of occwl edges, with edge ID as the key
    - edgeFace_IncM: Edge ID as the key, Adjacent faces ID as the value
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)

    ### Faces ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        extended_face = face
        face_dict[face_idx] = (face.surface_type(), face, extended_face)

    ### Edges and IncidenceMat ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue
        
        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge)
            edge_dict[edge_idx] = edge
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass  # ignore seam

    return face_dict, edge_dict, edgeFace_IncM


def get_vertice_from_edge(edge):
    vertice = []
    vert_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vert_explorer.More():
        vert = topods_Vertex(vert_explorer.Current())  
        vert_explorer.Next()
        vertice.append(vert)
    return vertice

def are_vertice_same(vertex1, vertex2, tolerance=1e-5):
    p1,p2 = vertex1,vertex2
    if not isinstance(vertex1,gp_Pnt): p1 = BRep_Tool.Pnt(vertex1)
    if not isinstance(vertex2,gp_Pnt): p2 = BRep_Tool.Pnt(vertex2)
    if p1.Distance(p2)>tolerance:
        return False
    return True

def are_connnect_vertice(vertice1,vertice2, tolerance=1e-5):
    for v1 in vertice1:
        for v2 in vertice2:
            if are_vertice_same(v1,v2,tolerance):
                return True
    return False


def judge_edges_connect(edges):
    all_group_edges = defaultdict(list)
    for edge in edges:
        vertice = get_vertice_from_edge(edge)
        need_connnect_group = []
        for group_idx,group_edges in all_group_edges.items():
            for now_group_edge in group_edges:
                vertice_now_group_edge = get_vertice_from_edge(now_group_edge)
                if are_connnect_vertice(vertice,vertice_now_group_edge):
                    need_connnect_group.append(group_idx)
                    break
        if len(need_connnect_group)==0:
            all_group_edges[len(all_group_edges)].append(edge)
        else:
            need_select_group_idx = min(need_connnect_group)
            all_group_edges[need_select_group_idx].append(edge)
            for idx in need_connnect_group:
                if idx != need_select_group_idx:
                    all_group_edges[need_select_group_idx].extend(all_group_edges[idx])
    return_edges = [edges for idx, edges in all_group_edges.items()]            
    return return_edges


def merge_edges(edges):
    
    if len(edges)==1:
        return edges[0]

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


def safe_uvgrid(face, method, num_u, num_v):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always") 
        try:
            result = uvgrid(face, method=method, num_u=num_u, num_v=num_v)
        except Exception as e:
            print(f"❌ uvgrid({method}) error: {e}")
            # traceback.print_exc()
            return None

        
        for w in wlist:
            if issubclass(w.category, RuntimeWarning):
                return None
        return result

def safe_ugrid(edge, method, num_u):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")  
        try:
            result = ugrid(edge, method=method, num_u=num_u)
        except Exception as e:
            print(f"❌ ugrid({method}) error: {e}")
            # traceback.print_exc()
            return None

        
        for w in wlist:
            if issubclass(w.category, RuntimeWarning):
                return None
        return result

def extract_primitive(solid, **kwargs):
    """
    Extract all primitive information from splitted solid

    Args:
    - solid (occwl.Solid): A single b-rep solid in occwl format

    Returns:
    - face_pnts (N x RAW_UV_GRID x RAW_UV_GRID x 3): Sampled uv-grid points on the bounded surface region (face)
    - face_masks (N x RAW_UV_GRID x RAW_UV_GRID x 1): Whether or not the sampled uv-grid points located inside the surface boundary
    - edge_pnts (M x RAW_UV_GRID x 3): Sampled u-grid points on the boundged curve region (edge)
    - edge_corner_pnts (M x 2 x 3): Start & end vertices per edge
    - edgeFace_IncM (M x 2): Edge-Face incident matrix, every edge is connect to two face IDs
    - faceEdge_IncM: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    assert isinstance(solid, Solid)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)

    # Skip unused index key, and update the adj
    face_dict, face_map = update_mapping(face_dict)
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM_update = {}
    for key, value in edgeFace_IncM.items():
        new_face_indices = [face_map[x] for x in value]
        edgeFace_IncM_update[edge_map[key]] = new_face_indices
    edgeFace_IncM = edgeFace_IncM_update
    

    ############################################# 
    try:
        reverse_edgeFace_IncM = defaultdict(list)
        for edge_index,[left_face_index,right_face_index] in edgeFace_IncM.items():
            reverse_edgeFace_IncM[tuple([left_face_index,right_face_index])].append(edge_index)  

        for k,edge_indice in reverse_edgeFace_IncM.items():
            if len(edge_indice)>=2:
                all_edges = [edge_dict[edge_indice[i]].topods_shape() for i in range(len(edge_indice))]
                all_group_edges = judge_edges_connect(all_edges)
                select_edge_indice = edge_indice[:len(all_group_edges)]
                del_edge_indice = edge_indice[len(all_group_edges):]
                for i,group_edges in enumerate(all_group_edges):
                    new_edge = merge_edges(group_edges)
                    edge_dict[select_edge_indice[i]] = Edge(new_edge)
                    edgeFace_IncM[select_edge_indice[i]] = list(k)
                for idx in del_edge_indice:
                    del edge_dict[idx]
                    del edgeFace_IncM[idx]
    except:
        return None
    ##################################################
    
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM, edgeface_map = update_mapping(edgeFace_IncM)

    #############################################
    edgeFace_IncM = np.stack([x for x in edgeFace_IncM.values()])

    # Face-edge adj
    num_faces = len(face_dict)
    faceEdge_IncM = []
    for surf_idx in range(num_faces):
        surf_edges, _ = np.where(edgeFace_IncM == surf_idx)
        faceEdge_IncM.append(surf_edges)

    # Sample uv-grid from surface (RAW_UV_GRIDxRAW_UV_GRID)
    graph_face_feat = {}
    face_types = {}
    for face_idx, face_feature in face_dict.items():
        _, face, extended_face = face_feature
        points = safe_uvgrid(
            extended_face, method="point", num_u=kwargs['RAW_UV_GRID'], num_v=kwargs['RAW_UV_GRID']
        )
        normals = safe_uvgrid(
            extended_face, method="normal", num_u=kwargs['RAW_UV_GRID'], num_v=kwargs['RAW_UV_GRID']
        )
        visibility_status = safe_uvgrid(
            face, method="visibility_status", num_u=kwargs['RAW_UV_GRID'], num_v=kwargs['RAW_UV_GRID']
        )
        
        # process special data
        if points is None or normals is None or visibility_status is None:
            return None

        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat[face_idx] = face_feat
        face_types[face_idx] = get_face_type(face)
        
    face_attrs = np.stack([x for x in graph_face_feat.values()])
    face_pnts = face_attrs[:, :, :, :3]
    face_normals = face_attrs[:, :, :, 3:6]
    face_masks = face_attrs[:, :, :, 6:]
    face_types = np.stack([x for x in face_types.values()])

    # sample u-grid from curve (1xRAW_UV_GRID)
    graph_edge_feat = {}
    graph_corner_feat = {}
    for edge_idx, edge in edge_dict.items():
        points = safe_ugrid(edge, method="point", num_u=kwargs['RAW_UV_GRID'])
        normals = safe_ugrid(edge, method="tangent", num_u=kwargs['RAW_UV_GRID'])
        edge_feat = np.concatenate((points, normals), axis=-1)
        graph_edge_feat[edge_idx] = edge_feat
        #### edge corners as start/end vertex ###
        v_start = points[0]
        v_end = points[-1]
        graph_corner_feat[edge_idx] = (v_start, v_end)
    edge_attrs = np.stack([x for x in graph_edge_feat.values()])
    edge_pnts = edge_attrs[..., :3]
    edge_normals = edge_attrs[..., 3:6]
    edge_corner_pnts = np.stack([x for x in graph_corner_feat.values()])

    return [face_pnts, face_normals, face_masks, edge_pnts, edge_normals, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM,face_types]

