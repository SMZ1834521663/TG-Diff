import igraph as ig
import numpy as np
import pickle
import torch
from multiprocessing import Pool
from tqdm import tqdm

# ============================================================
# numpy adj -> igraph Graph
# ============================================================
def adj_np_to_igraph(adj_np: np.ndarray) -> ig.Graph:
    """
    adj_np: (N, N) numpy array, 0/1
    """
    n = adj_np.shape[0]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if adj_np[i, j] != 0:
                edges.append((i, j))

    return ig.Graph(n=n, edges=edges, directed=False)


# ============================================================
# worker: canonical adjacency (numpy only)
# ============================================================
def topo_canonical_adj_np(adj_np: np.ndarray) -> np.ndarray:
    """
    Subprocess function
    """
    g = adj_np_to_igraph(adj_np)
    perm = g.canonical_permutation()
    return adj_np[perm][:, perm]


# ============================================================
# worker: canonical hash (numpy only, for dedup)
# ============================================================
def topo_canonical_hash_np(adj_np: np.ndarray) -> bytes:
    """
    返回 canonical adjacency 的 bytes hash
    """
    canon = topo_canonical_adj_np(adj_np)
    return canon.tobytes()


# ============================================================
# multiprocess topo processing
# ============================================================
def process_topos(
    train_adjs_np,
    num_workers=32,
    deduplicate: bool = False,
):
    """
    Parameters
    ----------
    train_adjs_np : list[np.ndarray]
    deduplicate : bool
        False -> only canonical（return list[np.ndarray]）
        True  -> canonical + deduplicate（return list[np.ndarray]）
    """

    with Pool(
        processes=num_workers,
        maxtasksperchild=100,  
    ) as p:

        if not deduplicate:
            # -------- only canonicalize --------
            results = list(
                tqdm(
                    p.imap(
                        topo_canonical_adj_np,
                        train_adjs_np,
                        chunksize=8,
                    ),
                    total=len(train_adjs_np),
                    desc="Canonicalizing",
                )
            )
            return results

        else:
            # -------- canonical + dedup --------
            hashes = list(
                tqdm(
                    p.imap(
                        topo_canonical_hash_np,
                        train_adjs_np,
                        chunksize=8,
                    ),
                    total=len(train_adjs_np),
                    desc="Canonical hashing",
                )
            )

    seen = set()
    unique_adjs = []

    for h, adj_np in zip(hashes, train_adjs_np):
        if h not in seen:
            seen.add(h)
            unique_adjs.append(adj_np)

    return unique_adjs


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    # ================== config ==================
    NUM_WORKERS = 32
    INPUT_PATH = "./topo/pkl/f050_furniture_all_topo_adj.pkl"
    DEDUPLICATE = False   
    if DEDUPLICATE == True:
        OUTPUT_PATH = "./topo/pkl/f050_furniture_train_topo_deduplicated.pkl"  #canonical  deduplicated
    else:
        OUTPUT_PATH = "./topo/pkl/f050_furniture_train_topo_canonical.pkl"  #canonical  deduplicated
    # ==========================================

    # 1. load torch adjacency list
    with open(INPUT_PATH, "rb") as f:
        train_adjs = pickle.load(f)  # list[torch.Tensor]

    print(f"original topo count: {len(train_adjs)}")

    # 2. torch -> numpy
    train_adjs_np = [
        adj.cpu().numpy().astype(np.uint8)
        for adj in train_adjs
    ]

    # 3. multiprocess process
    processed_adjs_np = process_topos(
        train_adjs_np,
        num_workers=NUM_WORKERS,
        deduplicate=DEDUPLICATE,
    )

    print(f"output topo count: {len(processed_adjs_np)}")

    # 4. numpy -> torch
    processed_adjs = [
        torch.from_numpy(adj_np)
        for adj_np in processed_adjs_np
    ]

    # 5. save
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(processed_adjs, f)

    print("✔ Done.")

