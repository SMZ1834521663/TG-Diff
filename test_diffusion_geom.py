import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vae_geom import VAE_Geom
from models.diffusion_geom import Diffusion_Geom
from data_process.dataset import SolidDataset
from utils.utils import data_to_device,load_config,load_checkpoints,permute_adj_list,pad_adj_matrices,seed_everything
from utils.ddpm_scheduler import SNR_DDPMScheduler


def main(args):
    # prepare
    # seed_everything(666)
    cfg = load_config(args.cfg_path)
    data_cfg, diffusion_cfg = cfg.data, cfg.model
    data_name = data_cfg.data_name
    test_data_path = data_cfg.path_test
    max_faces = data_cfg.max_faces

    device = "cuda"
    batch_size = args.batch_size
    use_real_topo = True

    #out dir
    exp_dir = os.path.join(diffusion_cfg.output_dir, diffusion_cfg.output_tag + "_" + data_name)
    save_dir = os.path.join(exp_dir, "test")
    os.makedirs(save_dir, exist_ok=True)

    # create dataset
    test_dataset = SolidDataset(test_data_path,cfg,mode="diffusion")
    print(f"\nLoad [{len(test_dataset)}] testing solids")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=diffusion_cfg.num_workers,
        pin_memory=False,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    #create model
    model = Diffusion_Geom(
        input_size=128,
        patch_size=1,
        hidden_dim=[384,768],
        depth=[6,2],
        num_heads=8,
        mlp_ratio=4,
    ).to(device)
    load_checkpoints(model,diffusion_cfg.pretrained_path,ema_states = None)

    # Load pretrained hvae 
    vae_f = VAE_Geom(
        in_channels = 3,
        mid_channels = 8,
        out_channels = 3,
        conv_norm_group = 4,
        attn_nhead = 8
    ).to(device)
    load_checkpoints(vae_f,diffusion_cfg.pretrained_vae_path)
    for param in vae_f.parameters():  #frezze
        param.requires_grad = False
    vae_f = vae_f.eval()

    # Initialize diffusion scheduler
    noise_scheduler = SNR_DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='v_prediction',
        clip_sample=False,
        rescale_betas_zero_snr=True,
        snr_min=0.03, 
        snr_max=1000.0,
        snr_power = 1
    )

    #######################################
    model.eval()
    if not use_real_topo:
        print("use gen topo")
        with open("deepcad_topology_only_N3000_30x30.pkl", "rb") as f:
            all_adj_matrix = pickle.load(f)
        all_adj_matrix = [torch.tensor(adj) for adj in all_adj_matrix]
        adj_list = all_adj_matrix[0: 0 + batch_size]
        print(len(adj_list))
        adj_list = permute_adj_list(adj_list)
        adj_matrix, face_mask = pad_adj_matrices(adj_list, max_n=max_faces)
        adj_matrix,face_mask = adj_matrix.to(device).bool().float(),face_mask.to(device)
    else:
        for iter, data in enumerate(test_loader):
            data = data_to_device(data,device)
            face_mask = data["face_mask"].bool()
            adj_matrix = data["adj_matrix"]
            # test
            if iter !=args.sample_batch:
                continue
            break

    with torch.no_grad():
        z_f = torch.randn((batch_size, max_faces, 128)).to(device)*(face_mask.unsqueeze(-1))
        for t in tqdm(noise_scheduler.timesteps):
            timesteps = t.expand(batch_size,).to(z_f.device)
            surf_pred = model(z_f,face_mask,adj_matrix,timesteps,train=False)
            surf_pred = surf_pred*(face_mask.unsqueeze(-1))
            z_f = noise_scheduler.step(surf_pred, t, z_f).prev_sample*(face_mask.unsqueeze(-1))
        recon_surf, recon_surf_mask = vae_f.decode(z_f,face_mask)
    
    #save
    os.makedirs(save_dir,exist_ok=True)
    np.save(save_dir+"/"+"faces.npy",recon_surf.detach().cpu().numpy())
    # np.save(save_dir+"/"+"faces_ori.npy",face_points.detach().cpu().numpy())
    np.save(save_dir+"/"+"faces_masks.npy",face_mask.detach().cpu().numpy())
    # np.save(save_dir+"/"+"faces_point_masks.npy",recon_surf_mask.detach().cpu().numpy())
    np.save(save_dir+"/"+"adj_matrix.npy",adj_matrix.detach().cpu().numpy())


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=5 python ./test_diffusion_f.py --cfg_path ./config/deepcad_f0_30.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/diffusion_geom_deepcad_f0_30.yaml')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sample_batch", type=int, default=1)
    args = parser.parse_args()
    main(args)

