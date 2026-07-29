import os
import pickle
import traceback
import shutil
import numpy as np
import networkx as nx
from tqdm import tqdm
import trimesh
from trimesh.sample import sample_surface
from multiprocessing.pool import Pool

from occwl.io import load_step
from occwl.solid import Solid
from occwl.entity_mapper import EntityMapper

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Solid
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface)

from TG_Diff_final.data_process.data_process_utils import extract_primitive,get_bbox,build_face_adjacency,build_fef_adjacency,get_face_type



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
    
def normalize(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization
    """
    # Global normalization to -1~1
    total_points = np.array(surf_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals) / 2
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [], [], [], []

    # Normalize corner
    corner_wcs = (corner_pnts - global_offset[np.newaxis, :]) / (global_scale * 0.5)

    # Normalize surface
    for surf_pnt in surf_pnts:
        # Normalize CAD to WCS
        surf_pnt_wcs = (surf_pnt - global_offset[np.newaxis, np.newaxis, :]) / (global_scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(surf_pnt_wcs.reshape(-1, 3), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1, 3), axis=0)
        local_offset = min_vals + (max_vals - min_vals) / 2
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis, np.newaxis, :]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)

    # Normalize edge
    for edge_pnt in edge_pnts:
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis, :]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1, 3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1, 3), axis=0)
        local_offset = min_vals + (max_vals - min_vals) / 2
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs


def parse_solid(solid):
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    if len(list(solid.faces())) > MAX_FACE:
        return None
    if len(list(solid.faces())) < MIN_FACE:
        return None
    if len(list(solid.edges())) > MAX_EDGE:
        return None
    if len(list(solid.edges())) < MIN_EDGE:
        return None
    
    # Extract all B-rep primitives and their adjacency information
    result = extract_primitive(solid, RAW_UV_GRID=RAW_UV_GRID)
    if result == None:
        return None
    face_pnts, surf_normals, surfs_wcs_masks, edge_pnts, edge_normals, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM, face_types = result

    # Normalize the CAD model, restrict the point scale into [-1, 1]
    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)

    # Remove duplicate and merge corners
    corner_wcs = np.round(corner_wcs, 4)
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1, 3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1, 3)
        else:
            # Check if it exists or not
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1, 3)], 0)

    # Edge-corner adjacency
    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    # Surface global bbox
    surf_bboxes = []
    for pnts in surfs_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        surf_bboxes.append(np.concatenate([min_point, max_point]))
    surf_bboxes = np.vstack(surf_bboxes)

    # exclude
    _, counts = np.unique(edgeFace_IncM, return_counts=True)
    if np.max(counts) > FILTER_EDGE_PER_FACE:
        return None

    # exclude narrow solid
    all_pts = surfs_wcs.reshape(-1,3)
    solid_min_point,solid_max_point = get_bbox(all_pts)
    lengths = solid_max_point - solid_min_point
    if lengths.min()==0:
        return None
    aspect_ratio = lengths.max() / lengths.min()
    if aspect_ratio > EXCLUDE_NARROW_SCALE:
        return None

    # Edge global bbox
    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        #exclude small edge
        dis = np.sum(np.square(max_point-min_point))
        if dis < EXCLUDE_SMALL_EDGE_SCALE:
            return None
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    # make adj_matrix
    ff_adj = build_face_adjacency(edgeFace_IncM)

    fef_adj = build_fef_adjacency(edgeFace_IncM)
    if fef_adj.max()>2:
        return None

    G = nx.from_numpy_array(ff_adj.astype(int))
    adj_gde = nx.floyd_warshall_numpy(G)
    all_paths = dict(nx.all_pairs_shortest_path(G))
    if np.isinf(adj_gde).any():
        return None

    # Convert to float32 to save space
    data = {
        'surf_wcs': surfs_wcs.astype(np.float32),
        'surf_wcs_masks': surfs_wcs_masks.astype(np.float32),
        'edge_wcs': edges_wcs.astype(np.float32),
        'surf_normals': surf_normals.astype(np.float32),
        'edge_normals': edge_normals.astype(np.float32),
        'corner_wcs': corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj': edgeCorner_IncM,
        'faceEdge_adj': faceEdge_IncM,       
        'surf_bbox_wcs': surf_bboxes.astype(np.float32),
        'edge_bbox_wcs': edge_bboxes.astype(np.float32),
        'corner_unique': corner_unique.astype(np.float32),
        "face_types":face_types.astype(np.int64),
        "ff_adj":ff_adj.astype(np.int64),
    }

    return data


def load_step_file(file):
    reader = STEPControl_Reader()
    status=reader.ReadFile(file)
    if status !=1:return None
    reader.TransferRoots()
    shape = reader.Shape(1)
    return shape

def process(uid):
    try:
        ############################ step
        if mode == "abc" or mode == "deepcad":
            step_base_dir = os.path.join(RAW_STEP_FOLDER, uid.split('.')[0][:4], uid.split('.')[0])
            step_file_path = os.path.join(step_base_dir, os.listdir(step_base_dir)[0])
        elif mode == "furniture":
            step_file_path = uid
        # use OCC load
        cad_solid = load_step_file(step_file_path)
        if not isinstance(cad_solid,TopoDS_Solid): return 0
        cad_solid = Solid(cad_solid)

        # use occwl load
        # cad_solid = load_step(step_file_path)
        # if len(cad_solid)!=1: 
        #     return 0 
        # cad_solid = cad_solid[0]

        data = parse_solid(cad_solid)
        if data is None:  return 0 

        ############################ stl
        # add pc
        if USE_PC:
            stl_base_dir = os.path.join(RAW_STL_FOLDER, uid.split('.')[0][:4], uid.split('.')[0])
            stl_file_path = os.path.join(stl_base_dir, os.listdir(stl_base_dir)[0])
            out_mesh = trimesh.load(stl_file_path)
            out_pc, _ = sample_surface(out_mesh, N_POINTS)
            if len(out_pc)!=N_POINTS: return 0
        else:
            out_pc = np.zeros((N_POINTS,3))
        data.update({"pc":out_pc})

        ############################ Save the parsed result 
        if mode == "deepcad" or mode == "abc":
            data_uid = step_file_path.split('/')[-2] 
            sub_folder = data_uid[:4]
            save_folder = os.path.join(OUTPUT, sub_folder)
            os.makedirs(save_folder,exist_ok=True)
            save_path = os.path.join(save_folder, data['uid']+'.pkl')
        elif mode == "furniture":
            data_uid = step_file_path.split('/')[-1].split(".")[0]
            save_path = os.path.join(OUTPUT, data_uid + '.pkl')

        data.update({"uid":data_uid})
        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)
        return 1 

    except Exception as e:
        # print(traceback.print_exc(e))
        print('not saving due to error...')
        return step_file_path

if __name__ == '__main__':
    # param config
    mode = "furniture"
    option = "train"
    RAW_UV_GRID = 16
    RAW_STEP_FOLDER = "/data/ybc2021/Datasets/ABC"
    RAW_STL_FOLDER = "/data/songhx24/Dataset/ABCSTL"

    if mode=="deepcad":
        MIN_FACE = 0
        MAX_FACE = 30
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 20
        EXCLUDE_NARROW_SCALE = 1000                   
        EXCLUDE_SMALL_EDGE_SCALE = 0.0000                 
        OUTPUT = '/data/ybc2021/Datasets/Work5_Data/deepcad_f030' + "/" + option
        DATA_NAME_LIST = '/data/ybc2021/Datasets/DeepCAD/deepcad_data_split_6bit.pkl'
        USE_PC = True
        N_POINTS = 8192
    elif mode == "abc":
        MIN_FACE = 0
        MAX_FACE = 50
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 30 
        EXCLUDE_NARROW_SCALE = 1000
        EXCLUDE_SMALL_EDGE_SCALE = 0.0000
        OUTPUT = '/data/ybc2021/Datasets/Work5_Data/abc_f050' + "/" + option
        DATA_NAME_LIST = '/data/ybc2021/Datasets/ABC/abc_data_split_6bit.pkl'
        USE_PC = True
        N_POINTS = 8192
    elif mode == "furniture":
        MIN_FACE = 0
        MAX_FACE = 50
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 30 
        EXCLUDE_NARROW_SCALE = 1000
        EXCLUDE_SMALL_EDGE_SCALE = 0.0000
        OUTPUT = '/data/ybc2021/Datasets/Work5_Data/furniture_f050' + "/" + option
        DATA_LIST = '/data/ybc2021/Datasets/Furniture/' + "/" + option
        USE_PC = False
        N_POINTS = 8192

    
    shutil.rmtree(OUTPUT,ignore_errors=True)
    os.makedirs(OUTPUT,exist_ok=True)

    # load_data
    if mode == "deepcad" or mode == "abc":
        with open(DATA_NAME_LIST, 'rb') as file:
            data_list = pickle.load(file)
        print("loading_data_dir")
        data_list_all = data_list[option] 
    elif mode == "furniture":
        data_list_all = [os.path.join(DATA_LIST, name) for name in os.listdir(DATA_LIST)]

    valid = 0
    convert_iter = Pool(1).imap_unordered(process, data_list_all,chunksize=4) 
    for data in tqdm(convert_iter, total=len(data_list_all)):
        if isinstance(data,int):
            valid += data 
    print(f'Done... Data Converted Ratio {100.0*valid/len(data_list_all)}%')



    
    





