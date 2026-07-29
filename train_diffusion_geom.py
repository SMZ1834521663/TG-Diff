import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from models.vae_geom import VAE_Geom
from models.diffusion_geom import Diffusion_Geom
from data_process.dataset import SolidDataset
from utils.utils import simple_logger,data_to_device,load_config,save_checkpoints,load_checkpoints,compute_snr_weights,seed_everything
from utils.ddpm_scheduler import SNR_DDPMScheduler,snr_based_beta_schedule
from utils.lr_scheduler import CosineWarmupLambda

def main(args):
    # prepare
    # seed_everything(666)
    cfg = load_config(args.cfg_path)
    data_cfg, diffusion_cfg = cfg.data, cfg.model
    data_name = data_cfg.data_name
    train_data_path = data_cfg.path_train
    max_faces = data_cfg.max_faces

    # create outdir
    exp_dir = os.path.join(diffusion_cfg.output_dir, diffusion_cfg.output_tag  + "_" + data_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(diffusion_cfg.output_dir,exist_ok=True)
    os.makedirs(exp_dir,exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # create dataset
    train_dataset = SolidDataset(train_data_path,cfg,mode="diffusion")
    print(f"\nLoad [{len(train_dataset)}] training solids")
    train_loader = DataLoader(
        train_dataset,
        batch_size=diffusion_cfg.batch_size,
        num_workers=diffusion_cfg.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    # create model
    parallel = diffusion_cfg.parallel
    device = "cuda" 
    model = Diffusion_Geom(
        input_size=128,
        patch_size=1,
        hidden_dim=[384,768],
        depth=[6,2],
        num_heads=8,
        mlp_ratio=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = diffusion_cfg.lr, betas=diffusion_cfg.betas, weight_decay = diffusion_cfg.weight_decay)
    lr_lambda = CosineWarmupLambda(
        one_epoch_step=len(train_loader),
        warm_up_epochs=100,
        cosine_epochs=100,
        decay_milestones=(1500, 3000, 4500),
        whole_ratios=(0.8, 0.6, 0.4),
        lr_min_ratio=0.1,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.last_epoch=len(train_loader)*diffusion_cfg.last_epoch-1
    scheduler.step()

    if diffusion_cfg.pretrained_path!="":
        load_checkpoints(model,diffusion_cfg.pretrained_path,optimizer=None,ema_states=None,strict=True)
    if parallel: model = nn.DataParallel(model) 

    # Load pretrained vae 
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

    #set weight for diffusion loss
    weights = compute_snr_weights(snr_based_beta_schedule(timesteps=1000,snr_min=0.03,snr_max=1000,snr_power=1)).to(device)

    #create logger
    log_name = "logs" + "_" + time.strftime("%Y%m%d") + ".txt" 
    logger = simple_logger(os.path.join(log_dir, log_name))

    # preloading vae data
    preload = True
    if preload:
        for iter, data in tqdm(enumerate(train_loader),ncols=200, ascii=True, dynamic_ncols=False):
            data = data_to_device(data,device)
            face_points = data["face_points"]
            face_points_mask =  data["face_points_mask"].float()
            face_mask = data["face_mask"].bool()
            adj_matrix = data["adj_matrix"].float()
            ef_adj = data["ef_adj"]
            B,LF = face_mask.shape
            # VAE data
            with torch.no_grad():
                params_f, _ = vae_f.encode(face_points, face_mask, face_points_mask, ef_adj)
                z_f, _ = vae_f.gaussian_sample(params_f,face_mask,mode = True)
            train_dataset.z_f[data["idx"]] = z_f.detach().cpu()
            train_dataset.adj_matrix[data["idx"]] = adj_matrix.detach().cpu()
        train_dataset.diffusion_add_over=True

    # start
    for epoch in range(diffusion_cfg.epochs):
        epoch = epoch + 1
        # train
        model.train()
        steps_per_epoch = len(train_loader)
        progress_bar = tqdm(total=steps_per_epoch,ncols=200, ascii=True, dynamic_ncols=False)
        progress_bar.set_description(f"train epoch {epoch}")
        accu_loss = torch.zeros((1))  
        for iter, data in enumerate(train_loader):
            data = data_to_device(data,device)
            face_points = data["face_points"]
            face_points_mask =  data["face_points_mask"].float()
            face_mask = data["face_mask"].bool()
            adj_matrix = data["adj_matrix"].float()
            ef_adj = data["ef_adj"]
            B,LF = face_mask.shape

            if not preload:
                with torch.no_grad():
                    params_f, _ = vae_f.encode(face_points, face_mask, face_points_mask, ef_adj)
                    z_f, _ = vae_f.gaussian_sample(params_f,face_mask,mode = True)
            else:
                z_f = data["z_f"]
                adj_matrix = data["adj_matrix"]
            
            optimizer.zero_grad()
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,)).to(device).long()
            surf_noise = torch.randn(z_f.shape).to(device)*(face_mask.unsqueeze(-1))
            z_f_diffused = noise_scheduler.add_noise(z_f, surf_noise, timesteps)*(face_mask.unsqueeze(-1))
            target_v = noise_scheduler.get_velocity(z_f, surf_noise, timesteps)
            loss_surf = model(z_f_diffused,face_mask,adj_matrix,timesteps,target_v=target_v,train=True)
            loss_surf = (loss_surf* weights[timesteps]).mean()
            
            total_loss = loss_surf*diffusion_cfg.loss_weight

            total_loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=50.0)
            optimizer.step()
            scheduler.step()

            current_lrs = [group['lr'] for group in optimizer.param_groups]

            # progress_bar
            accu_loss+=torch.tensor([loss_surf.detach()]) 
            progress_bar.update(1)
            progress_bar.set_postfix({
                "f": f"{accu_loss[0].item() / (iter + 1):.6f}",
                "lr": f"{current_lrs[0] :.6f}",
            })

        #log
        if epoch %diffusion_cfg.log_per_epoch == 0:
            logger.log(epoch, accu_loss/(iter+1))
        
        # save
        if epoch % diffusion_cfg.save_per_epoch == 0:
            print("saved:{}".format(epoch))
            filename = data_name +"_" + "geom_diff"  +"_" + "epoch_{:05d}.pth".format(epoch)
            save_checkpoints(model, ckpt_dir, optimizer=None, filename=filename, parallel=parallel)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1,2,3  python ./train_diffusion_geom.py --cfg_path ./config/diffusion_geom_abc_f0_50.yaml
    # CUDA_VISIBLE_DEVICES=0,2,3  python ./train_diffusion_geom.py --cfg_path ./config/diffusion_geom_deepcad_f7_30.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/deepcad_f7_30.yaml')
    args = parser.parse_args()
    main(args)


           
