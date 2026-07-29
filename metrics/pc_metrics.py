import torch 
import os
import numpy as np
from tqdm import tqdm
import random
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData
from pathlib import Path
import multiprocessing
import argparse
from chamferdist import ChamferDistance
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods

from pc_sample import N_POINTS,load_data_with_suffix

class Step_Collecter():
    def __init__(self,folder_path):
        self.folder_path = folder_path

    def collect_step_files(self,folder_path):
        step_paths = []
        for suffix in (".step", ".stp"):
            step_paths.extend(load_data_with_suffix(folder_path, suffix))
        return sorted(step_paths)


    def read_step_shape(self,step_path):
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            raise ValueError("cannot read STEP file: {}".format(step_path))
        if not reader.TransferRoots():
            raise ValueError("cannot transfer STEP roots: {}".format(step_path))
        return reader.OneShape()


class PC_Collecter():
    def __init__(self,folder_path):
        self.folder_path = folder_path 

    def read_ply(self,path):
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)
            x = np.array(plydata['vertex']['x'])
            y = np.array(plydata['vertex']['y'])
            z = np.array(plydata['vertex']['z'])
            vertex = np.stack([x, y, z], axis=1)
        return vertex

    def downsample_pc(self, points, n):
        sample_idx = random.sample(list(range(points.shape[0])), n)
        return points[sample_idx]

    def normalize_pc(self,points):
        # normalize
        mean = np.mean(points, axis=0)
        points = (points - mean) 
        # fit to unit cube
        scale = np.max(np.abs(points))  
        points = points / scale
        return points

    def collect_pc(self,path): #process once
        pc = self.read_ply(path)
        if pc.shape[0] > N_POINTS:
            pc = self.downsample_pc(pc, N_POINTS)
        pc = self.normalize_pc(pc)
        return pc
    
    def process(self):
        pcs = []
        paths = []
        shape_paths = load_data_with_suffix(self.folder_path, '.ply')
        with multiprocessing.Pool(32) as pool:
            load_iter = pool.imap_unordered(self.collect_pc, shape_paths)
            for pc in tqdm(load_iter, total=len(shape_paths)):
                if len(pc) > 0:
                    pcs.append(pc)
        pcs = np.stack(pcs, axis=0)
        print("point clouds: {}".format(pcs.shape))
        return pcs
    

class CC_Computer():
    def __init__(self):
        pass

    def find_parent(self,parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_parent(self,parent, a, b):
        ra = self.find_parent(parent, a)
        rb = self.find_parent(parent, b)
        if ra != rb:
            parent[rb] = ra

    def compute_once(self,step_path):
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return 0
        if not reader.TransferRoots():
            return 0
        shape = reader.OneShape()

        vertex_map = TopTools_IndexedMapOfShape()
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_VERTEX, vertex_map)
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        parent = list(range(vertex_map.Size() + 1))
        active_vertices = set()
        num_edges = 0

        for edge_idx in range(1, edge_map.Size() + 1):
            edge = topods.Edge(edge_map.FindKey(edge_idx))
            if BRep_Tool.Degenerated(edge):
                continue

            first_idx = vertex_map.FindIndex(topexp.FirstVertex(edge))
            last_idx = vertex_map.FindIndex(topexp.LastVertex(edge))
            if first_idx == 0 or last_idx == 0:
                continue

            num_edges += 1
            active_vertices.add(first_idx)
            active_vertices.add(last_idx)
            if first_idx != last_idx:
                self.union_parent(parent, first_idx, last_idx)

        num_vertices = len(active_vertices)
        num_components = len({self.find_parent(parent, idx) for idx in active_vertices})
        return int(max(num_edges - num_vertices + 2 * num_components, 0))
    

class COV_MMD_Computer():
    def __init__(self):
        pass

    def _pairwise_CD(self, sample_pcs, ref_pcs, batch_size):
        N_sample = sample_pcs.shape[0]
        N_ref = ref_pcs.shape[0]
        all_cd = []
        all_emd = []
        iterator = range(N_sample)
        matched_gt = []
        device = sample_pcs.device.index
        pbar = tqdm(iterator,disable=False,leave=True)
        chamfer_dist = ChamferDistance()

        for sample_b_start in pbar:
            sample_batch = sample_pcs[sample_b_start]

            cd_lst = []
            emd_lst = []
            for ref_b_start in range(0, N_ref, batch_size):
                ref_b_end = min(N_ref, ref_b_start + batch_size)
                ref_batch = ref_pcs[ref_b_start:ref_b_end]

                batch_size_ref = ref_batch.size(0)
                sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
                sample_batch_exp = sample_batch_exp.contiguous()
                
                cd = chamfer_dist(sample_batch_exp,ref_batch,batch_reduction=None, point_reduction="mean",bidirectional=True)
                cd_lst.append(cd.view(1, -1))

            cd_lst = torch.cat(cd_lst, dim=1)
            all_cd.append(cd_lst)

            hit = np.argmin(cd_lst.detach().cpu().numpy()[0])
            matched_gt.append(hit)
            pbar.set_postfix({"cov": len(np.unique(matched_gt)) * 1.0 / N_ref})

        all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

        return all_cd


    def compute_once(self, sample_pcs, ref_pcs, batch_size): 
        all_dist = self._pairwise_CD(sample_pcs, ref_pcs, batch_size)
        N_sample, N_ref = all_dist.size(0), all_dist.size(1)
        min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
        min_val, _ = torch.min(all_dist, dim=0)
        mmd = min_val.mean()
        cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
        cov = torch.tensor(cov).to(all_dist)

        return cov.item(), mmd.item()
    

class JSD_Computer():
    def __init__(self):
        pass

    def _jsdiv(self, P, Q):
        '''another way of computing JSD'''

        def _kldiv(A, B):
            a = A.copy()
            b = B.copy()
            idx = np.logical_and(a > 0, b > 0)
            a = a[idx]
            b = b[idx]
            return np.sum([v for v in a * np.log2(a / b)])

        P_ = P / np.sum(P)
        Q_ = Q / np.sum(Q)

        M = 0.5 * (P_ + Q_)

        return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))
    
    def unit_cube_grid_point_cloud(self,resolution, clip_sphere=False):
        '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
        that is placed in the unit-cube.
        If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
        '''
        grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
        spacing = 1.0 / float(resolution - 1) * 2
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    grid[i, j, k, 0] = i * spacing - 0.5 * 2
                    grid[i, j, k, 1] = j * spacing - 0.5 * 2
                    grid[i, j, k, 2] = k * spacing - 0.5 * 2

        if clip_sphere:
            grid = grid.reshape(-1, 3)
            grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

        return grid, spacing
    
    def entropy_of_occupancy_grid(self,pclouds, grid_resolution, in_sphere=False):
        '''Given a collection of point-clouds, estimate the entropy of the random variables
        corresponding to occupancy-grid activation patterns.
        Inputs:
            pclouds: (numpy array) #point-clouds x points per point-cloud x 3
            grid_resolution (int) size of occupancy grid that will be used.
        '''
        epsilon = 10e-4
        bound = 1 + epsilon
        if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
            print(abs(np.max(pclouds)), abs(np.min(pclouds)))
            warnings.warn('Point-clouds are not in unit cube.')

        if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
            warnings.warn('Point-clouds are not in unit sphere.')

        grid_coordinates, _ = self.unit_cube_grid_point_cloud(grid_resolution, in_sphere)
        grid_coordinates = grid_coordinates.reshape(-1, 3)
        grid_counters = np.zeros(len(grid_coordinates))
        grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
        nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

        for pc in pclouds:
            _, indices = nn.kneighbors(pc)
            indices = np.squeeze(indices)
            for i in indices:
                grid_counters[i] += 1
            indices = np.unique(indices)
            for i in indices:
                grid_bernoulli_rvars[i] += 1

        acc_entropy = 0.0
        n = float(len(pclouds))
        for g in grid_bernoulli_rvars:
            p = 0.0
            if g > 0:
                p = float(g) / n
                acc_entropy += entropy([p, 1.0 - p])

        return acc_entropy / len(grid_counters), grid_counters

    def jensen_shannon_divergence(self, P, Q):
        if np.any(P < 0) or np.any(Q < 0):
            raise ValueError('Negative values.')
        if len(P) != len(Q):
            raise ValueError('Non equal size.')

        P_ = P / np.sum(P)  # Ensure probabilities.
        Q_ = Q / np.sum(Q)

        e1 = entropy(P_, base=2)
        e2 = entropy(Q_, base=2)
        e_sum = entropy((P_ + Q_) / 2.0, base=2)
        res = e_sum - ((e1 + e2) / 2.0)

        res2 = self._jsdiv(P_, Q_)

        if not np.allclose(res, res2, atol=10e-5, rtol=0):
            warnings.warn('Numerical values of two JSD methods don\'t agree.')

        return res
    
    def compute_once(self, sample_pcs, ref_pcs, in_unit_sphere, resolution=28):
        '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
        Args:
            sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
            ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
            resolution: (int) grid-resolution. Affects granularity of measurements.
        '''
        sample_grid_var = self.entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
        ref_grid_var = self.entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
        return self.jensen_shannon_divergence(sample_grid_var, ref_grid_var)

##########################################################################################
# cov mmd jsd
cc_computer = CC_Computer()
cov_mmd_computer = COV_MMD_Computer()
jsd_computer = JSD_Computer()

def run_one_iter(i, sample_pcs, ref_pcs, n_test, batch_size, device_id):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    select_idx = random.sample(range(len(sample_pcs)), 3 * int(n_test))
    rand_sample_pcs = sample_pcs[select_idx]
    select_idx = random.sample(range(len(ref_pcs)), int(n_test))
    rand_ref_pcs = ref_pcs[select_idx]

    jsd = jsd_computer.compute_once(rand_sample_pcs, rand_ref_pcs, in_unit_sphere=False)
    with torch.no_grad():
        rand_sample_pcs = torch.tensor(rand_sample_pcs, device=device)
        rand_ref_pcs = torch.tensor(rand_ref_pcs, device=device)
        cov, mmd = cov_mmd_computer.compute_once(rand_sample_pcs,rand_ref_pcs, batch_size=batch_size)

    return {"COV": cov, "MMD": mmd, "JSD": jsd}

def _worker(args):
    return run_one_iter(*args)

def run_multi_gpu(
    n_times,
    sample_pcs,
    ref_pcs,
    n_test,
    batch_size,
    gpu_ids=(0),
):
    ctx = multiprocessing.get_context("spawn")
    results = []
    tasks = []
    for i in range(n_times):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((i, sample_pcs, ref_pcs, n_test, batch_size, gpu_id))

    with ctx.Pool(processes=len(gpu_ids)) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            results.append(res)
    return results

def main(args):
    real_path = args.real_path
    fake_path = args.fake_path
    ################################### Load ply 
    pc_collecter_fake = PC_Collecter(fake_path)
    pc_collecter_real = PC_Collecter(real_path)
    sample_pcs = pc_collecter_fake.process()
    ref_pcs = pc_collecter_real.process()
    
    result_list = []
    n_times = 3
    n_test = 1000
    batch_size = 64

    result_list = run_multi_gpu(
        n_times=n_times,
        sample_pcs=sample_pcs,
        ref_pcs=ref_pcs,
        n_test=n_test,
        batch_size=batch_size,
        gpu_ids=[0],
    )
    for res in result_list:
        print(res)
    avg_result = {}
    for k in result_list[0].keys():
        vals = [x[k] for x in result_list]
        avg_result.update({"avg-" + k: float(np.mean(vals))})
        avg_result.update({"var-" + k: float(np.var(vals))})
    print("average/variance result:")
    print(avg_result)

    ################################### Load step
    # step_collecter = Step_Collecter(fake_path)
    # step_paths = step_collecter.collect_step_files(fake_path)
    # cc_values = []   
    # with multiprocessing.Pool(processes=32) as pool:
    #     iterator = pool.imap_unordered(cc_computer.compute_once, step_paths)
    #     for cc in tqdm(iterator, total=len(step_paths), desc="Computing STEP CC"):
    #         cc_values.append(cc)
    # print("avg-CC:", float(np.mean(cc_values)))
    ##################################### avg metrics
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, default="/data/ybc2021/Projects/work5_makedata/ori_deepcad_f7302_test_uv")
    parser.add_argument("--fake_path", type=str, default="/data/ybc2021/Projects/work5_makedata/2_500/exp")
    args = parser.parse_args()
    main(args)
