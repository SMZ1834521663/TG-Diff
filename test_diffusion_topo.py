import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import networkx as nx
import torch
import pickle
from tqdm import tqdm

from models.diffusion_topo import Diffusion_Topo
from utils.utils import seed_everything,data_to_device,load_config,load_checkpoints
from utils.d3pm_scheduler import LOG_ZERO,COS_D3PMScheduler,log_onehot_to_index,index_to_log_onehot
from utils.topo_distribution_ratio import ratio_all_7_30,ratio_all_0_30,ratio_all_0_50


def extract_adj(x, N, offset=-1,move_zero_row = True, two_2_one = True):  #[B, N*(N-1)//2], values in {0,1,2}
    B, L = x.shape
    device = x.device
    dtype = x.dtype

    # ---------- 1. Lower Triangle Index ----------
    row_idx, col_idx = torch.tril_indices(N, N, offset=offset, device=device)

    # ---------- 2. row statistics ----------
    is2 = (x == 2)
    is1 = (x == 1)

    row_len = torch.zeros(N, device=device, dtype=torch.long)
    row_len.scatter_add_(0, row_idx, torch.ones_like(row_idx))

    cnt2 = torch.zeros(B, N, device=device, dtype=torch.long)
    cnt1 = torch.zeros(B, N, device=device, dtype=torch.long)
    cnt2.scatter_add_(1, row_idx.unsqueeze(0).expand(B, -1), is2.long())
    cnt1.scatter_add_(1, row_idx.unsqueeze(0).expand(B, -1), is1.long())

    # ---------- 3. padding row candidate ----------
    cond_all_2 = cnt2 == row_len.unsqueeze(0)
    cond_major_2 = cnt2 * 2 > row_len.unsqueeze(0)
    cond_no_1 = cnt1 == 0
    pad_candidate = cond_all_2 | cond_major_2 | cond_no_1
    pad_candidate[:, 0] = False 

    # ---------- 4. continuous padding rows at the end ----------
    rev = pad_candidate.flip(dims=[1])
    suffix = torch.cumprod(rev.long(), dim=1).bool()
    pad_row_mask = suffix.flip(dims=[1])

    # ---------- 5. construct an effective row mask ----------
    mask = ~pad_row_mask  # True = valid
    pad_mask_expand = (~mask).unsqueeze(-1) | (~mask).unsqueeze(-2)

    # ---------- 6. construct adjacency matrix ----------
    adj = torch.zeros(B, N, N, dtype=dtype, device=device)
    if two_2_one:
        tri_valid = mask[:, row_idx] & mask[:, col_idx]   # [B, L]
        x = torch.where((x == 2) & tri_valid, torch.ones_like(x), x)

    valid = x != 2
    adj[:, row_idx, col_idx] = x * valid
    adj[:, col_idx, row_idx] = x * valid
    # padding
    adj[pad_mask_expand] = 0
    # diagonal reset
    adj.diagonal(dim1=1, dim2=2).zero_()

    # ---------- 7. special processing ----------
    if move_zero_row:
        is_zero_row = (adj.abs().sum(dim=-1) < 2)
        pad_row = ~mask | is_zero_row
        sort_idx = torch.argsort(pad_row.to(torch.int), dim=1)

        # sort adj 
        adj = torch.gather(adj, dim=1, index=sort_idx.unsqueeze(-1).expand(-1, -1, N))
        adj = torch.gather(adj, dim=2, index=sort_idx.unsqueeze(1).expand(-1, N, -1))

        # sort mask
        mask = torch.gather(mask, dim=1, index=sort_idx)
        mask[adj.abs().sum(dim=-1) < 2] = False 
        pad_mask_expand = (~mask).unsqueeze(-1) | (~mask).unsqueeze(-2) 
        adj[pad_mask_expand] = 0

    return mask, adj

def extract_unpadded_adj(adj, node_mask): # (B, N, N)  (B, N)
    adj_list = []
    B = adj.shape[0]
    for b in range(B):
        k = int(node_mask[b].sum())
        if k == 0:
            continue
        adj_k = adj[b, :k, :k].cpu()
        adj_list.append(adj_k)

    return adj_list  # List[Tensor(k, k)]


def check_topo_connect(adj_list):
    connect_count = 0
    for to in adj_list:
        adj = to.detach().cpu().numpy() # N x N
        G = nx.from_numpy_array(adj)
        is_connected = nx.is_connected(G)
        if is_connected:
            connect_count+=1
    print("connect:",connect_count/len(adj_list))


def main(args):
    # prepare
    # seed_everything(666)
    cfg = load_config(args.cfg_path)
    data_cfg, diffusion_cfg = cfg.data, cfg.model
    data_name = data_cfg.data_name
    max_faces = data_cfg.max_faces

    device = "cuda"
    batch_size = args.batch_size
    mode = args.mode

    #out dir
    exp_dir = os.path.join(diffusion_cfg.output_dir, diffusion_cfg.output_tag + "_" + data_name)
    save_dir = os.path.join(exp_dir, "test")
    os.makedirs(save_dir, exist_ok=True)
    
    #create model
    model = Diffusion_Topo(
        hidden_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4,
        edge_emb_dim = 8,
        edge_num_classes = 2,
        max_faces = max_faces
    ).to(device)
    load_checkpoints(model,diffusion_cfg.pretrained_path,ema_states = None,strict=True)

    # Initialize diffusion scheduler
    d3pm_scheduler = COS_D3PMScheduler(
        num_train_timesteps=200,
        prediction_type = 'x0',
        num_classes = 2
    )

    #########################################################
    model.eval()
    if mode == "deepcad_f7_30":
        probs = ratio_all_7_30
        min_num, max_num = 7,30
    if mode == "deepcad_f0_30":
        probs = ratio_all_0_30
        min_num, max_num = 0,30
    if mode == "abc_f0_50":
        probs = ratio_all_0_50
        min_num, max_num = 0,50
        
    face_num = torch.multinomial(probs,num_samples=batch_size,replacement=True) + min_num
    idx = torch.arange(max_num, device=device) 
    face_mask = (idx.unsqueeze(0) < face_num.to(device).unsqueeze(1)).bool().to(device)
    B,LF = batch_size,max_num
    # start
    with torch.no_grad():
        zero_logits_x = torch.zeros((B, 3, LF*(LF-1)//2), device=device) 
        one_logits_x = torch.ones((B, 1, LF*(LF-1)//2),device=device)  
        mask_logits_x = torch.cat([zero_logits_x, one_logits_x], dim=1)
        log_x_t = torch.log(mask_logits_x).clamp(min=LOG_ZERO)  # (B, Co, N)
        log_x_t_class = log_x_t     # (B, num_classes+1, N)

        for t in tqdm(range(d3pm_scheduler.num_timesteps - 1, -1, -1)):
            t = torch.tensor(t).to(device)
            timesteps = t.expand(batch_size,).to(device)
            x_t_class = log_onehot_to_index(log_x_t_class) #B,C,N->B,N
            e_class_pred = model( 
                x_t_class,
                timesteps,
                mask = face_mask,
                train=False
            )
            # deterministic sampling
            if mode == "deepcad_f7_30":
                scale = (1 + 0.8* (1 - t / 200)+0.25)   #730  
            if mode == "deepcad_f0_30":
                scale = (1 + 0.25* (1 - t / 200)+0.1)   #030
            if mode == "abc_f0_50":
                scale = (1 + 1.5* (1 - t / 200)+0.36)   #050

            log_x_recon = d3pm_scheduler.log_pred_from_denoise_out(e_class_pred*scale)
            if t == 0:
                log_EV_qxt_x0 = log_x_recon
            else:
                log_EV_qxt_x0 = d3pm_scheduler.q_posterior(log_x_start=log_x_recon, log_x_t=log_x_t_class, t=timesteps)

            log_x_t_class = d3pm_scheduler.log_sample_categorical(log_EV_qxt_x0)

        # topo post process
        recon_adj_flatten = log_onehot_to_index(log_x_t_class[:, :-1, ...]) #B,N*(N+1)
        f_mask, adj_clean = extract_adj(recon_adj_flatten,LF,offset=-1,move_zero_row=True)
        same_per_batch = (face_mask == f_mask).all(dim=1)
        ratio = same_per_batch.float().mean().item()
        print("mask sucess:",ratio)
        mask_mean = f_mask.sum(dim=1).float().mean()
        print("mean_num:",mask_mean)
        adj_list = extract_unpadded_adj(adj_clean,f_mask)
        print("save num",len(adj_list))
        check_topo_connect(adj_list)

        #save
        with open("./gen_topo_adj.pkl", "wb") as f:
            pickle.dump(adj_list, f)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=5 python ./test_diffusion_f.py --cfg_path ./config/diffusion_fm_new_mix_deepcad_f7_30.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/diffusion_topo_deepcad_f0_30.yaml')  #diffusion_topo_abc_f0_50
    parser.add_argument("--mode", type=str, default="deepcad_f0_30") #deepcad_f0_30 deepcad_f7_30 abc_f0_50
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    main(args)

