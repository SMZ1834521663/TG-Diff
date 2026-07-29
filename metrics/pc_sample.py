import os
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import trimesh
from trimesh.sample import sample_surface
from plyfile import PlyData, PlyElement
import numpy as np
import argparse

#########
N_POINTS = 2000

def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])

def load_data_with_suffix(root_folder, suffix):
    data_files = []
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(suffix):
                file_path = os.path.join(root, filename).replace("\\","/")
                data_files.append(file_path)

    return data_files

# core
class SamplePoints:
    def __init__(self,in_dir,out_dir):
        self.in_dir = in_dir  
        self.out_dir = out_dir

    def write_ply(self, points, filename, text=False):
        """ input: Nx3, write points to filename as PLY format. """
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        with open(filename, mode='wb') as f:
            PlyData([el], text=text).write(f)

    def process_once(self, path):
        filename = path.split('/')[-1].split(".")[0]
        out_mesh = trimesh.load(path)
        # out_mesh.apply_scale(3.0)
        out_pc, _ = sample_surface(out_mesh, N_POINTS)
        save_path = os.path.join(self.out_dir + "/" + filename + '.ply')
        self.write_ply(out_pc, save_path)
        return

    def process(self):
        os.makedirs(self.out_dir,exist_ok=True)
        shape_paths = load_data_with_suffix(self.in_dir, '.stl')
        with multiprocessing.Pool(8) as pool: 
            convert_iter =  pool.imap_unordered(self.process_once, shape_paths) #multiprocessing.cpu_count()
            for _ in tqdm(convert_iter, total=len(shape_paths)):
                pass
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="/data/ybc2021/Projects/work5_makedata/2_500/exp")
    args = parser.parse_args()
    in_dir = args.in_dir 
    out_dir = in_dir
    sp = SamplePoints(in_dir,out_dir) 
    sp.process()
