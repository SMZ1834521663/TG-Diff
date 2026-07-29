import json
import pickle
import numpy as np
import h5py
import os
import signal
from tqdm import tqdm
from hashlib import sha256

def real2bit(data, n_bits=6, min_range=-1.0, max_range=1.0):
    range_quantize = 2 ** n_bits - 1
    data_q = (data - min_range) * range_quantize / (max_range - min_range)
    data_q = np.clip(data_q, 0, range_quantize)
    return data_q.astype(np.int32)

def cad_hash_from_surf_attr(surf_attr, bit):
    xyz = surf_attr[:, :3]
    xyz_q = real2bit(xyz, n_bits=bit)
    return sha256(xyz_q.tobytes()).hexdigest()


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def worker_ABC_pkl2h5(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    (
        surf_wcs, 
        surf_wcs_masks, 
        edge_wcs, 
        surf_normals, 
        edge_normals, 
        corner_wcs, 
        edgeface_adj, 
        edgeCorner_adj, 
        faceEdge_adj, 
        surf_bbox_wcs, 
        edge_bbox_wcs, 
        corner_unique,
        face_types,
        ff_adj,
        pc,
        data_uid
    ) = data.values()

    num_face, num_edge, num_corner = surf_wcs.shape[0], edge_wcs.shape[0], corner_unique.shape[0]
    idx_in_chunk = np.ones([max(num_face, num_edge, num_corner), 1])
    surf_attr = np.concatenate([surf_wcs, surf_normals, surf_wcs_masks], axis=-1).reshape((num_face, -1))
    surf_attr = np.concatenate([
                                surf_attr, 
                                surf_bbox_wcs.reshape(num_face, -1),
                                face_types.reshape(num_face, -1), 
                                idx_in_chunk[:num_face, :]], axis=-1)

    edge_attr = np.concatenate([edge_wcs,edge_normals], axis=-1).reshape((num_edge, -1))
    edge_attr = np.concatenate([
                                edge_attr, 
                                edge_bbox_wcs.reshape(num_edge, -1), 
                                corner_wcs.reshape(num_edge, -1),
                                edgeCorner_adj, 
                                edgeface_adj, 
                                idx_in_chunk[:num_edge, :]], axis=-1)

    corner_attr = np.concatenate([corner_unique, idx_in_chunk[:num_corner, :]], axis=-1)

    ff_adj_pad = np.zeros((MAX_FACE,MAX_FACE))
    ff_adj_pad[:len(ff_adj),:len(ff_adj)] = ff_adj

    if ADD_PC:
        global_attr = np.concatenate([pc.reshape(1,-1), ff_adj_pad.reshape(1,-1), np.array([1]).reshape(1,1)], axis=-1) # 8192,3  30,30  30,30
    else:
        global_attr = np.concatenate([ff_adj_pad.reshape(1,-1), np.array([1]).reshape(1,1)], axis=-1)

  
    _, counts = np.unique(edgeface_adj, return_counts=True)

    float_type = np.float16 if FP16 else np.float32
    return surf_attr.astype(float_type),\
            edge_attr.astype(float_type), \
            corner_attr.astype(float_type),\
            global_attr.astype(float_type), \
            data_uid[:8],\
            counts


def load_valid_abc_pkl(pkl_root_folder, pkl_name_list):
    with open(pkl_name_list, 'rb') as f:
        abc_data = pickle.load(f)
        train, val, test = abc_data['train'], abc_data['val'], abc_data['test']
        train_uid, val_uid, test_uid = set([uid.split('.')[0] for uid in train]), \
            set([uid.split('.')[0] for uid in val]), \
            set([uid.split('.')[0] for uid in test])

    full_uids = []
    dirs = [f'{pkl_root_folder}/{str(i).zfill(4)}' for i in range(100)]
    for folder in dirs:
        files = os.listdir(folder)
        full_uids += [file.split('.')[0] for file in files]

    valid_train = list(
        map(lambda x: x + '.pkl', list(set(full_uids).intersection(train_uid)))
    )
    valid_val = list(
        map(lambda x: x + '.pkl', list(set(full_uids).intersection(val_uid)))
    )
    valid_test = list(
        map(lambda x: x + '.pkl', list(set(full_uids).intersection(test_uid)))
    )
    return valid_train, valid_val, valid_test

def convert_pkl2h5(save_h5_path, datalist):
    """
    
    The 0th dimension of chunks represents the number of chunks
    surface (surf_wcs + surf_ncs + surf_masks + surf_wcs bbox + face_type + surf_idx in chunks)；
        chunks = [?, FACE_CHUNK, RAW_UV_GRID * RAW_UV_GRID * 7 + 6 + 1 + 1]
    edge (edge_wcs + edge_ncs + edge_wcs_bbox + corner_wcs + ef_adj + ec_adj + edge_idx in chunks)；
        chunks = [?, EDGE_CHUNK, RAW_UV_GRID * 6 + 6 + 6 + 2 + 2 + 1]
    corner (corner_unique + corner_idx in chunks)
        chunks = [?, CORNER_CHUNK, 3 + 1]
    global (pc + adj_matrix + adj_gde in chunks)
        chunks = [?, GLOBAL_CHUNK, 8192*3+30*30+30*30]
    data_uid (data_name, in_which_chunk, related_position within the chunk(0, 1, 2, ...))

    Process logic, sequentially pass the pkl files to workr_ABC_pkl2h5, ensuring that the total number of faces in the pkl files passed in at once does not exceed Face-CHANK, and the total number of curves does not exceed EDGE-CHANK
    """
    
    src = list(map(lambda x: x, datalist))
    results = map(worker_ABC_pkl2h5, src)

    dataset_shape_inc = 1000 #int(len(src)/5)
    chl_surf_attr = RAW_UV_GRID * RAW_UV_GRID * 7 + 6 + 1 + 1
    chl_edge_attr = RAW_UV_GRID * 6 + 6 + 6 + 2 + 2 + 1
    chl_corner_attr = 3 + 1
    if ADD_PC:
        chl_global_attr = N_POINTS*3 + MAX_FACE*MAX_FACE + 1
    else:
        chl_global_attr = MAX_FACE*MAX_FACE + 1
    surf_attr_ls, edge_attr_ls, corner_attr_ls, global_attr_ls = [], [], [], []  
    surf_attr_chk, edge_attr_chk, corner_attr_chk, global_attr_chk, data_uid_chk = [], [], [], [], [] 
    float_type = np.float16 if FP16 else np.float32
    seen_cad_hash = set()
    with h5py.File(save_h5_path, 'w') as f:
        surf_dataset = f.create_dataset(
            'surf_attr', (dataset_shape_inc, FACE_CHUNK, chl_surf_attr), dtype=float_type,
            maxshape=(None, FACE_CHUNK, chl_surf_attr), chunks=(1, FACE_CHUNK, chl_surf_attr)
        )
        edge_dataset = f.create_dataset(
            'edge_attr', (dataset_shape_inc, EDGE_CHUNK, chl_edge_attr), dtype=float_type,
            maxshape=(None, EDGE_CHUNK, chl_edge_attr), chunks=(1, EDGE_CHUNK, chl_edge_attr)
        )
        corner_dataset = f.create_dataset(
            'corner_attr', (dataset_shape_inc, CORNER_CHUNK, chl_corner_attr), dtype=float_type, 
            maxshape=(None, CORNER_CHUNK, chl_corner_attr),chunks=(1, CORNER_CHUNK, chl_corner_attr)
        )
        global_dataset = f.create_dataset(
            'global_attr', (dataset_shape_inc, GLOBAL_CHUNK, chl_global_attr), dtype=float_type, 
            maxshape=(None, GLOBAL_CHUNK, chl_global_attr),chunks=(1, GLOBAL_CHUNK, chl_global_attr)
        )

        # init
        current_index = 0  
        surf_chunk_len, edge_chunk_len, corner_chunk_len,global_chunk_len = 0, 0, 0, 0  # How much data is currently stored inside each chunk;
        for res in tqdm(results, total=len(src)):
            surf_attr, edge_attr, corner_attr, global_attr, data_uid, edge_per_face_counts = res

            # if surf_attr.shape[0] > MAX_FACE :continue
            # if surf_attr.shape[0] < MIN_FACE :continue
            # if edge_attr.shape[0] > MAX_EDGE :continue
            # if edge_attr.shape[0] < MAX_EDGE :continue
            # if np.max(edge_per_face_counts) > FILTER_EDGE_PER_FACE: continue
            if USE_DEDUPLICATE:
                cad_hash = cad_hash_from_surf_attr(surf_attr, bit=6)
                if cad_hash in seen_cad_hash:
                    continue 
                seen_cad_hash.add(cad_hash)

            if surf_chunk_len + surf_attr.shape[0] > FACE_CHUNK or \
                edge_chunk_len + edge_attr.shape[0] > EDGE_CHUNK or \
                corner_chunk_len + corner_attr.shape[0] > CORNER_CHUNK or\
                global_chunk_len + 1> GLOBAL_CHUNK:

                """Padding and concatenating the attributes in the list; Write into h5py"""
                surf_pad, edge_pad, corner_pad, global_pad = -1 * np.ones([FACE_CHUNK - surf_chunk_len, chl_surf_attr]), \
                                                            -1 * np.ones([EDGE_CHUNK - edge_chunk_len, chl_edge_attr]), \
                                                            -1 * np.ones([CORNER_CHUNK - corner_chunk_len, chl_corner_attr]),\
                                                            -1 * np.ones([GLOBAL_CHUNK - global_chunk_len, chl_global_attr])
                surf_attr_ls.append(surf_pad)
                edge_attr_ls.append(edge_pad)
                corner_attr_ls.append(corner_pad)
                global_attr_ls.append(global_pad)

                surf_attr_chk.append(np.concatenate(surf_attr_ls, axis=0)[np.newaxis, ...])
                edge_attr_chk.append(np.concatenate(edge_attr_ls, axis=0)[np.newaxis, ...])
                corner_attr_chk.append(np.concatenate(corner_attr_ls, axis=0)[np.newaxis, ...])
                global_attr_chk.append(np.concatenate(global_attr_ls, axis=0)[np.newaxis, ...])

                # Reset the surf_attr_ls state and fill in the current surf_attr
                surf_attr[:, -1] *= 0
                edge_attr[:, -1] *= 0
                corner_attr[:, -1] *= 0
                global_attr[:, -1] *= 0
                data_uid = data_uid + '_' + str(current_index + len(surf_attr_chk)) + '_0'

                surf_attr_ls, edge_attr_ls, corner_attr_ls, global_attr_ls = [surf_attr], [edge_attr], [corner_attr], [global_attr]
                data_uid_chk.append(np.array([data_uid]).astype('S20'))

                surf_chunk_len = surf_attr.shape[0]
                edge_chunk_len = edge_attr.shape[0]
                corner_chunk_len = corner_attr.shape[0]
                global_chunk_len = 1

                if current_index + len(surf_attr_chk) >= surf_dataset.shape[0]:
                    # Write into H5 dataset
                    new_size = max(surf_dataset.shape[0] + dataset_shape_inc, current_index + len(surf_attr_chk))
                    surf_dataset.resize((new_size, FACE_CHUNK, chl_surf_attr))
                    edge_dataset.resize((new_size, EDGE_CHUNK, chl_edge_attr))
                    corner_dataset.resize((new_size, CORNER_CHUNK, chl_corner_attr))
                    global_dataset.resize((new_size, GLOBAL_CHUNK, chl_global_attr))

                    surf_dataset[current_index:current_index + len(surf_attr_chk)] = \
                        np.concatenate(surf_attr_chk, axis=0)
                    edge_dataset[current_index:current_index + len(edge_attr_chk)] = \
                        np.concatenate(edge_attr_chk, axis=0)
                    corner_dataset[current_index:current_index + len(corner_attr_chk)] = \
                        np.concatenate(corner_attr_chk, axis=0)
                    global_dataset[current_index:current_index + len(global_attr_chk)] = \
                        np.concatenate(global_attr_chk, axis=0)

                    # Reset surf_attr_chk status
                    current_index = current_index + len(surf_attr_chk)
                    surf_attr_chk, edge_attr_chk, corner_attr_chk, global_attr_chk = [], [], [], []

            else:
                """Update surf.attr_ls, etc"""
                idx_in_chunk = len(surf_attr_ls)
                surf_chunk_len += surf_attr.shape[0]
                edge_chunk_len += edge_attr.shape[0]
                corner_chunk_len += corner_attr.shape[0]
                global_chunk_len += global_attr.shape[0]

                surf_attr[:, -1] *= idx_in_chunk
                edge_attr[:, -1] *= idx_in_chunk
                corner_attr[:, -1] *= idx_in_chunk
                global_attr[:, -1] *= idx_in_chunk

                data_uid = data_uid + '_' + str(current_index + len(surf_attr_chk)) + '_' + str(idx_in_chunk)

                data_uid_chk.append(np.array([data_uid]).astype('S20'))
                surf_attr_ls.append(surf_attr)
                edge_attr_ls.append(edge_attr)
                corner_attr_ls.append(corner_attr)
                global_attr_ls.append(global_attr)

        
        """Padding the remaining part of surf_ attr_ls and filling it into surf_ attr_chk, then adding the remaining results from surf_ attr_chk to the H5 dataset"""
        if len(surf_attr_ls) > 0:
            surf_pad, edge_pad, corner_pad,global_pad = -1 * np.ones([FACE_CHUNK - surf_chunk_len, chl_surf_attr]), \
                                                        -1 * np.ones([EDGE_CHUNK - edge_chunk_len, chl_edge_attr]), \
                                                        -1 * np.ones([CORNER_CHUNK - corner_chunk_len, chl_corner_attr]),\
                                                        -1 * np.ones([GLOBAL_CHUNK - global_chunk_len, chl_global_attr])
            surf_attr_ls.append(surf_pad)
            edge_attr_ls.append(edge_pad)
            corner_attr_ls.append(corner_pad)
            global_attr_ls.append(global_pad)

            surf_attr_chk.append(np.concatenate(surf_attr_ls, axis=0)[np.newaxis, ...])
            edge_attr_chk.append(np.concatenate(edge_attr_ls, axis=0)[np.newaxis, ...])
            corner_attr_chk.append(np.concatenate(corner_attr_ls, axis=0)[np.newaxis, ...])
            global_attr_chk.append(np.concatenate(global_attr_ls, axis=0)[np.newaxis, ...])

        if len(surf_attr_chk) > 0:
            surf_dataset[current_index:current_index + len(surf_attr_chk)] = \
                np.concatenate(surf_attr_chk, axis=0)
            edge_dataset[current_index:current_index + len(edge_attr_chk)] = \
                np.concatenate(edge_attr_chk, axis=0)
            corner_dataset[current_index:current_index + len(corner_attr_chk)] = \
                np.concatenate(corner_attr_chk, axis=0)
            global_dataset[current_index:current_index + len(global_attr_chk)] = \
                np.concatenate(global_attr_chk, axis=0)
            current_index = current_index + len(surf_attr_chk)

        # Determine the final 0th dimensional size of surf_data, etc
        surf_dataset.resize((current_index, FACE_CHUNK, chl_surf_attr))
        edge_dataset.resize((current_index, EDGE_CHUNK, chl_edge_attr))
        corner_dataset.resize((current_index, CORNER_CHUNK, chl_corner_attr))
        global_dataset.resize((current_index, GLOBAL_CHUNK, chl_global_attr))

        # write in data_uid_dataset
        data_uid_dataset = f.create_dataset('KEY', dtype=np.dtype('S20'), data=np.concatenate(data_uid_chk))


if __name__ == '__main__':
    mode = "abc"
    option = "train"
    RAW_UV_GRID = 16
    N_POINTS = 8192
    ADD_PC = False
    FP16 = True
    USE_DEDUPLICATE = True
    FACE_CHUNK = 500  
    EDGE_CHUNK = 1500
    CORNER_CHUNK = 1500
    GLOBAL_CHUNK = 50

    if mode=="deepcad":
        MIN_FACE = 0
        MAX_FACE = 30
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 20 
        PKL_FOLDER_PATH = '/data/ybc2021/Datasets/Work5_Data/deepcad_f030' + "/" + option
        H5_FILE_PREFIX = "f030"
    elif mode == "abc":
        MIN_FACE = 0
        MAX_FACE = 50
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 30 
        PKL_FOLDER_PATH = '/data/ybc2021/Datasets/Work5_Data/abc_f050'+ "/" + option
        H5_FILE_PREFIX = "f050"
    elif mode == "furniture":
        MIN_FACE = 0
        MAX_FACE = 50
        MIN_EDGE = 0
        MAX_EDGE = 2000
        FILTER_EDGE_PER_FACE = 30 
        PKL_FOLDER_PATH = '/data/ybc2021/Datasets/Work5_Data/furniture_f050' + "/" + option
        H5_FILE_PREFIX = "f050"

    # load_data
    if mode == "deepcad" or mode == "abc":
        datalist = []
        dirs = [f'{PKL_FOLDER_PATH}/{str(i).zfill(4)}' for i in range(100)]
        for folder in dirs:
            files = os.listdir(folder)
            datalist += [os.path.join(PKL_FOLDER_PATH, file[:4], file) for file in files]  #pkl_root_folder, x[:4]
    elif mode == "furniture":
        datalist =  [os.path.join(PKL_FOLDER_PATH, name) for name in os.listdir(PKL_FOLDER_PATH)]

    save_h5_path = PKL_FOLDER_PATH + "/" + "{}_{}.h5".format(H5_FILE_PREFIX,option)
    convert_pkl2h5(save_h5_path=save_h5_path, datalist=datalist)

