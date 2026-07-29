import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from models.diffusion_topo import Diffusion_Topo
from data_process.dataset import TopoDataset
from utils.utils import simple_logger,data_to_device,load_config,save_checkpoints,load_checkpoints,compute_snr_weights,seed_everything
from utils.d3pm_scheduler import COS_D3PMScheduler,log_onehot_to_index,index_to_log_onehot
from utils.ddpm_scheduler import snr_based_beta_schedule
from utils.lr_scheduler import CosineWarmupLambda

def main(args):
    # prepare
    # seed_everything(666)
    cfg = load_config(args.cfg_path)
    data_cfg, diffusion_cfg = cfg.data, cfg.model
    data_name = data_cfg.data_name
    train_data_path = data_cfg.path_train_pkl
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
    train_dataset = TopoDataset(pkl_path = train_data_path, max_face=max_faces)
    print(f"\nLoad [{len(train_dataset)}] training solids")
    train_loader = DataLoader(
        train_dataset,
        batch_size=diffusion_cfg.batch_size,
        num_workers=diffusion_cfg.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # create model
    parallel = diffusion_cfg.parallel
    device = "cuda"
    model = Diffusion_Topo(
        hidden_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4,
        edge_emb_dim = 8,
        edge_num_classes = 2,   
        max_faces = max_faces,
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
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    lr_scheduler.last_epoch=len(train_loader)*diffusion_cfg.last_epoch-1
    lr_scheduler.step()
    
    if diffusion_cfg.pretrained_path!="":
        load_checkpoints(model,diffusion_cfg.pretrained_path,optimizer=None,ema_states=None,strict=True)
    if parallel: model = nn.DataParallel(model) 

    # Initialize diffusion scheduler
    d3pm_scheduler = COS_D3PMScheduler(
        num_train_timesteps=200,
        prediction_type = 'x0',
        num_classes = 2
    )

    #set weight for diffusion loss
    weights = compute_snr_weights(snr_based_beta_schedule(timesteps=200,snr_min=0.03,snr_max=1000,snr_power=1)).to(device)

    #create logger
    log_name = "logs" + "_" + time.strftime("%Y%m%d") + ".txt" 
    logger = simple_logger(os.path.join(log_dir, log_name))

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
            face_mask = data["face_mask"].bool()
            adj_matrix = data["adj_matrix"].long()
            B,LF = face_mask.shape
            
            optimizer.zero_grad()
            timesteps = torch.randint(0, d3pm_scheduler.num_timesteps, (B,)).to(device).long()
            mask2d = face_mask.unsqueeze(1) & face_mask.unsqueeze(2)
            adj_matrix[~mask2d] = 2
            idx = torch.tril_indices(LF, LF, offset=-1, device=device)
            adj_flatten = adj_matrix[:, idx[0], idx[1]] 
            log_x_start = index_to_log_onehot(adj_flatten, 2 + 1 + 1)
            log_x_t = d3pm_scheduler.q_sample(log_x_start=log_x_start, t=timesteps)  
            x_t_class = log_onehot_to_index(log_x_t)

            loss_edge = model(  
                x_t_class,
                timesteps,
                face_mask,
                log_x_t,
                log_x_start,
                d3pm_scheduler,
                train=True
            )

            loss_edge = (loss_edge*weights[timesteps]).mean() 
            total_loss = loss_edge*diffusion_cfg.loss_weight 
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=50.0)
            optimizer.step()
            lr_scheduler.step()

            current_lrs = [group['lr'] for group in optimizer.param_groups]

            # progress_bar
            accu_loss+=torch.tensor([loss_edge.detach()]) 
            progress_bar.update(1)
            progress_bar.set_postfix({
                "e": f"{accu_loss[0].item() / (iter + 1):.6f}",
                "lr": f"{current_lrs[0] :.6f}",
            })

        #log
        if epoch %diffusion_cfg.log_per_epoch == 0:
            logger.log(epoch, accu_loss/(iter+1))
        
        # save
        if epoch % diffusion_cfg.save_per_epoch == 0:
            print("saved:{}".format(epoch))
            filename = data_name +"_" + "topo_diff" + "_" + "epoch_{:05d}.pth".format(epoch) 
            save_checkpoints(model, ckpt_dir, optimizer=None, filename=filename, parallel=parallel)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,3  python ./train_diffusion_topo.py --cfg_path ./config/diffusion_topo_abc_f0_50.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/diffusion_topo_deepcad_f7_30.yaml')
    args = parser.parse_args()
    main(args)


           
