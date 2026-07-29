import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vae_geom import VAE_Geom
from data_process.dataset import SolidDataset
from utils.utils import seed_everything,data_to_device,load_config,load_checkpoints

def main(args):
    # prepare
    # seed_everything(666)
    cfg = load_config(args.cfg_path)
    data_cfg,vae_cfg = cfg.data,cfg.model
    data_name = data_cfg.data_name
    test_data_path = data_cfg.path_test

    device = "cuda"
    batch_size=args.batch_size

    #out dir
    exp_dir = os.path.join(vae_cfg.output_dir, vae_cfg.output_tag + "_" + data_name)
    save_dir = os.path.join(exp_dir, "test")
    os.makedirs(save_dir, exist_ok=True)

    # create dataset
    test_dataset = SolidDataset(test_data_path,cfg,mode="vae")
    print(f"\nLoad [{len(test_dataset)}] testing solids")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=vae_cfg.num_workers,
        pin_memory=False,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # create model
    model = VAE_Geom(
        in_channels = 3,
        mid_channels = 8,
        out_channels = 3,
        conv_norm_group = 4,
        attn_nhead = 8,
        use_mmd=vae_cfg.use_mmd
    ).to(device)
    load_checkpoints(model,vae_cfg.pretrained_path,strict=False)

    # test
    with torch.no_grad():
        model.eval()
        steps_per_epoch = len(test_loader)
        progress_bar = tqdm(total=steps_per_epoch)
        progress_bar.set_description(f"test")
        accu_loss = torch.zeros((4))  
        for iter, data in enumerate(test_loader):
            # compute loss
            if args.test_loss== False and iter < args.sample_batch:
                continue
            data = data_to_device(data,device)
            face_points = data["face_points"]
            face_points_mask =  data["face_points_mask"].float()
            face_mask = data["face_mask"].bool()
            ef_adj = data["ef_adj"]
            adj_matrix = data["adj_matrix"]

            if args.test_loss:
                points_loss_f, points_mask_loss_f, kl_loss_f,mmd_loss_f = model(face_points, 
                                                                                face_mask, 
                                                                                face_points_mask, 
                                                                                ef_adj, 
                                                                                train=True) 
                points_loss_f,points_mask_loss_f = points_loss_f.mean(),points_mask_loss_f.mean()
                if kl_loss_f.ndimension()>0: kl_loss_f,mmd_loss_f = kl_loss_f.mean(),mmd_loss_f.mean()

                accu_loss += torch.tensor([points_loss_f.detach(),points_mask_loss_f.detach(),kl_loss_f.detach(),mmd_loss_f.detach()])
                # progress_bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "f": f"{accu_loss[0].item() / (iter + 1):.7f}",
                    "m": f"{accu_loss[1].item() / (iter + 1):.7f}",
                    "kl": f"{accu_loss[2].item() / (iter + 1):.5f}",
                    "mmd": f"{accu_loss[3].item() / (iter + 1):.5f}",
                })

            else:
                rec_face_points,rec_face_points_mask  = model(face_points, 
                                                                face_mask, 
                                                                face_points_mask, 
                                                                ef_adj, 
                                                                train=False) 
                        
            if iter == args.sample_batch and args.test_loss==False:
                np.save(save_dir+"/"+"faces_vae.npy",rec_face_points.detach().cpu().numpy())
                np.save(save_dir+"/"+"faces_masks_vae.npy",face_mask.detach().cpu().numpy())
                np.save(save_dir+"/"+"adj_matrix_vae.npy",adj_matrix.detach().cpu().numpy())
                break


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python ./test_vae_geom.py --cfg_path ./config/vae_geom_abc_f0_50.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/vae_geom_deepcad_f0_30.yaml')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_batch", type=int, default=0)
    parser.add_argument("--test_loss", type=bool, default=True)
    args = parser.parse_args()
    main(args)

