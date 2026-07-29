import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_process.dataset import SolidDataset
from utils.utils import load_config

def collect_all_adj_from_path(data_path, cfg):
    adj_ds = SolidDataset(data_path, cfg)

    loader = DataLoader(
        adj_ds,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=adj_ds.collate_fn,
        pin_memory=False,
    )

    all_adj = []
    for data in tqdm(loader, desc=f"Collecting from {data_path}"):
        length = len(data["adj_matrix"])
        for i in range(length):
            now_adj_matrix = data["adj_matrix"][i]
            now_face_mask = data["face_mask"][i]
            adj = now_adj_matrix[:torch.sum(now_face_mask),:torch.sum(now_face_mask)]
            assert not torch.any(torch.all(adj == 0, dim=1))
            all_adj.append(adj.bool())
    return all_adj

def main():
    cfg = load_config("./config/diffusion_topo_furniture_f0_50.yaml")
    data_cfg = cfg["data"]

    all_adj = []

    # # ========= train =========
    all_adj.extend(collect_all_adj_from_path(data_cfg["path_train"], cfg))

    # # ========= val =========
    # all_adj.extend(collect_all_adj_from_path(data_cfg["path_val"], cfg))

    # # ========= test =========
    # all_adj.extend(collect_all_adj_from_path(data_cfg["path_test"], cfg))

    # save_path = "./topo/pkl/f730_all_topo_adj.pkl"
    save_path = "./topo/pkl/f050_furniture_all_topo_adj.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_adj, f)

    print(f"✔ Done. Saved {len(all_adj)} topo in total.")


if __name__ == "__main__":
    main()
