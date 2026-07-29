import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import rotate_points


class SolidDataset(Dataset):
    def __init__(self, h5_data_path, cfg, mode="vae"):
        super().__init__()
        self.h5_data_path = h5_data_path
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self.max_faces = data_cfg.max_faces
        self.max_edges = data_cfg.max_edges
        self.raw_uv_grid = data_cfg.raw_uv_grid
        
        with h5py.File(h5_data_path, 'r') as f:
            self.data_uid = f['KEY'][:]

        #dynamic add
        self.diffusion_add_over = False
        self.vae_add_over = False
        if mode=="vae":
            self.use_aug = model_cfg.use_aug
            
        elif mode=="diffusion":
            self.use_aug = model_cfg.use_aug  
            self.vae_latent_dim = data_cfg.vae_latent_dim
            self.z_f = torch.zeros((len(self.data_uid), self.max_faces, self.vae_latent_dim))
            self.adj_matrix = torch.zeros((len(self.data_uid), self.max_faces, self.max_faces))   #maybe it is useless

    def __len__(self):
        return len(self.data_uid)

    def _load_solid_data(self, chunk_id, id_in_chunk):
        """
        :param chunk_id: Locate the index of the h5py data chunk where the current data is located
        :param id_in_chunk:  After locating the chunk, the index of the current data within that chunk
        :return:
        """
        # TODO: load and parse h5 data;
        RAW_UV_GRID = self.raw_uv_grid
        face_attr, edge_attr,corner_attr,global_attr = self.face_attr[chunk_id], self.edge_attr[chunk_id],self.corner_attr[chunk_id],self.global_attr[chunk_id]
        
        face = face_attr[face_attr[:, -1] == id_in_chunk].astype(np.float32)
        edge = edge_attr[edge_attr[:, -1] == id_in_chunk].astype(np.float32)
        corner = corner_attr[corner_attr[:, -1] == id_in_chunk].astype(np.float32)
        global_data = global_attr[global_attr[:, -1] == id_in_chunk].astype(np.float32)

        face_uv, face_bbox, face_types, face_idx, _ = np.split(
            face, axis=1, indices_or_sections=[
                RAW_UV_GRID * RAW_UV_GRID * 7, 
                RAW_UV_GRID * RAW_UV_GRID * 7 + 6, 
                RAW_UV_GRID * RAW_UV_GRID * 7 + 7, 
                RAW_UV_GRID * RAW_UV_GRID * 7 + 8
            ]
        )
        face_uv = face_uv.reshape((-1, RAW_UV_GRID, RAW_UV_GRID, 7)).transpose((0, 3, 1, 2))
        face_wcs = face_uv[:, :3, ...]
        face_normals = face_uv[:, 3:6, ...]
        face_wcs_mask = face_uv[:, 6:, ...]

        edge_u, edge_bbox, corner_per, ec_adj, ef_adj, edge_idx, _ = np.split(
            edge, axis=1, indices_or_sections=[
                RAW_UV_GRID * 6, 
                RAW_UV_GRID * 6 + 6, 
                RAW_UV_GRID * 6 + 12,
                RAW_UV_GRID * 6 + 14, 
                RAW_UV_GRID * 6 + 16, 
                RAW_UV_GRID * 6 + 17
            ]
        )
        edge_u = edge_u.reshape((-1, RAW_UV_GRID, 6)).transpose((0, 2, 1))
        edge_wcs = edge_u[:, :3, :] 
        edge_normals = edge_u[:, 3:, :] 

        corner_crd, corner_idx, _ = np.split(corner, axis=1, indices_or_sections=[3, 4])  

        adj_matrix_pad, fef_pad, adj_gde_pad, global_idx, _ = np.split(
            global_data, axis=1, indices_or_sections=[
                
                self.max_faces*self.max_faces,
                self.max_faces*self.max_faces*2,
                self.max_faces*self.max_faces*3,
                self.max_faces*self.max_faces*3+1
            ]
        )  

        adj_matrix_pad = adj_matrix_pad.reshape(self.max_faces,self.max_faces)
        fef_pad = fef_pad.reshape(self.max_faces,self.max_faces)
        adj_gde_pad = adj_gde_pad.reshape(self.max_faces,self.max_faces)

        return torch.from_numpy(face_wcs).to(torch.float32), \
                torch.from_numpy(edge_wcs).to(torch.float32), \
                torch.from_numpy(ef_adj).to(torch.int64),\
                torch.from_numpy(face_normals).to(torch.float32),\
                torch.from_numpy(edge_normals).to(torch.float32),\
                torch.from_numpy(face_wcs_mask).to(torch.float32),\
                torch.from_numpy(face_types).to(torch.int64),\
                torch.from_numpy(adj_matrix_pad).to(torch.int64),\
                torch.from_numpy(fef_pad).to(torch.int64),\
                torch.from_numpy(adj_gde_pad).to(torch.int64),\

    
    def __getitem__(self, idx):
        #delay read
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_data_path, 'r')
            f = self.h5_file
            self.face_attr = f['surf_attr']
            self.edge_attr = f['edge_attr']
            self.corner_attr = f['corner_attr']
            self.global_attr = f['global_attr']

        # get raw data
        data_uid = str(self.data_uid[idx].astype(str)).split('_')
        chunk_id, id_in_chunk = int(data_uid[1]), int(data_uid[2])
        # [LF,3,16,16] [LE,3,16] [LV,3]   [LE,2]  [LE,2]    [LF,1]   [LE,1]    [LV,1]
        surf_wcs, edge_wcs, ef_adj, face_normals, edge_normals, face_wcs_mask,face_types,adj_matrix_pad,fef_pad,adj_gde_pad = self._load_solid_data(chunk_id, id_in_chunk)

        # aug 
        if self.use_aug:
            RAW_UV_GRID = self.raw_uv_grid
            if np.random.rand()>0.6:
                for axis in ['x', 'y', 'z']:
                    angle = np.random.choice([90, 180, 270])
                    angle = torch.tensor(angle)
                    pts = surf_wcs.permute(0,2,3,1).reshape(-1,3)
                    pts_rot = rotate_points(pts, angle, axis)
                    surf_wcs = pts_rot.reshape(-1, RAW_UV_GRID, RAW_UV_GRID, 3).permute(0,3,1,2)
     

        sample = {
                    "idx":idx,
                    'filename': data_uid[0],
                    "ef_adj":ef_adj,
                    "face_points":surf_wcs,
                    "face_points_mask":face_wcs_mask,
                    "adj_matrix_pad":adj_matrix_pad,
                }
        if self.diffusion_add_over:
            sample.update({"z_f":self.z_f[idx]})
            sample.update({"adj_matrix":self.adj_matrix[idx]})

        if self.vae_add_over:
            pass

        return sample
    
    def collate_fn(self, samples): 
      
        samples_data_dense = {
            "idx":[],
            "filename": [],
            "ef_adj":[],
            "face_points":[],
            "face_mask":[],
            "face_points_mask":[],
            "adj_matrix":[],
        }
        add_data_dense_diffusion = {
            "z_f":[],
            "adj_matrix":[],
        }
        add_data_dense_vae = {
        }

        max_faces = self.max_faces
        max_edges = self.max_edges

        for sample in samples:
            samples_data_dense["idx"].append(sample["idx"])
            samples_data_dense["filename"].append(sample["filename"])

            ef_adj_pad = -1* np.ones((max_edges,2),dtype=np.int64)       #-1 is ef_adj padding
            ef_adj_pad[:sample["ef_adj"].shape[0]]=sample["ef_adj"]
            samples_data_dense["ef_adj"].append(ef_adj_pad)

            face_points_pad=np.zeros((max_faces,3, self.raw_uv_grid, self.raw_uv_grid),dtype=np.float32)
            face_points_pad[:sample["face_points"].shape[0]]=sample["face_points"]
            samples_data_dense["face_points"].append(face_points_pad)

            face_mask_pad = np.zeros(max_faces, dtype=np.int64)
            face_mask_pad[:len(sample["face_points"])] = 1
            samples_data_dense["face_mask"].append(face_mask_pad)

            face_points_mask_pad=np.zeros((max_faces,1,self.raw_uv_grid,self.raw_uv_grid),dtype=np.int64)
            face_points_mask_pad[:sample["face_points_mask"].shape[0]]=sample["face_points_mask"]
            samples_data_dense["face_points_mask"].append(face_points_mask_pad)

            samples_data_dense["adj_matrix"].append(sample["adj_matrix_pad"])

            if self.diffusion_add_over:
                z_f_pad=np.zeros((max_faces,self.vae_latent_dim),dtype=np.float32)
                z_f_pad[:sample["z_f"].shape[0]]=sample["z_f"]
                add_data_dense_diffusion["z_f"].append(z_f_pad)

                adj_matrix_pad=np.zeros((max_faces,max_faces),dtype=np.float32)
                adj_matrix_pad[:sample["adj_matrix"].shape[0]]=sample["adj_matrix"]
                add_data_dense_diffusion["adj_matrix"].append(adj_matrix_pad)
                
            if self.vae_add_over:
                pass

        if self.diffusion_add_over:
            samples_data_dense.update({"z_f":add_data_dense_diffusion["z_f"]})
            samples_data_dense.update({"adj_matrix":add_data_dense_diffusion["adj_matrix"]})

        if self.vae_add_over:
            pass

        for k, v in samples_data_dense.items():
            if k =="face_mask" or k=="face_points_mask" or k=="adj_matrix" or k=="ef_adj":
                samples_data_dense[k] = torch.from_numpy(np.stack(v, axis=0)).long()
            elif k =="filename" or k=="idx":
                pass
            else:
                samples_data_dense[k] = torch.from_numpy(np.stack(v, axis=0))

        return samples_data_dense


class TopoDataset(Dataset):
    def __init__(self,pkl_path: str,max_face: int):
        with open(pkl_path, "rb") as f:
            self.adjs = pickle.load(f)
        self.max_face = max_face

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        return self.adjs[idx]

    def collate_fn(self, batch):
        samples_data_dense = {
            "adj_matrix": [],  # (B, max_face, max_face)
            "face_mask": []    # (B, max_face)
       
        }
        B = len(batch)
        device = batch[0].device
        dtype = batch[0].dtype
        adj_pad = torch.zeros(B, self.max_face, self.max_face,dtype=dtype,device=device)
        mask = torch.zeros(B, self.max_face,dtype=torch.bool,device=device)

        for i, adj in enumerate(batch):
            n = adj.shape[0]
            assert n <= self.max_face, f"n={n} > max_face={self.max_face}"
            adj_pad[i, :n, :n] = adj
            mask[i, :n] = True

        samples_data_dense.update({"adj_matrix":adj_pad})
        samples_data_dense.update({"face_mask":mask})    
        return samples_data_dense
    
if __name__ == "__main__":
    pass
    # from utils.prepare_utils import load_config
    # cfg = load_config("/data/songhx24/brepgen_twostep/config/vae_geom_abc_f0_50.yaml")
    # dataset = SolidGraphDataset("/data/songhx24/dataset_smz/abc/f050_filted_train.h5",cfg)
    # print(len(dataset))