import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from models.vae_geom import VAE_Geom
from data_process.dataset import SolidDataset
from utils.utils import data_to_device,initialize_weights,load_config,simple_logger,load_checkpoints,save_checkpoints,seed_everything
from utils.lr_scheduler import CosineWarmupLambda

def main(args):
    # prepare
    # seed_everything(666) 
    cfg = load_config(args.cfg_path)
    data_cfg,vae_cfg = cfg.data,cfg.model
    data_name = data_cfg.data_name
    train_data_path = data_cfg.path_train

    # create outdir
    exp_dir = os.path.join(vae_cfg.output_dir, vae_cfg.output_tag + "_" + data_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(vae_cfg.output_dir,exist_ok=True)
    os.makedirs(exp_dir,exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # create dataset
    train_dataset = SolidDataset(train_data_path,cfg,mode="vae")
    print(f"\nLoad [{len(train_dataset)}] training solids")
    train_loader = DataLoader(
        train_dataset,
        batch_size=vae_cfg.batch_size,
        num_workers=vae_cfg.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # create model
    parallel = vae_cfg.parallel
    if parallel: device = "cuda" 
    else: device = "cuda" 
    model = VAE_Geom(
        in_channels = 3,
        mid_channels = 8,
        out_channels = 3,
        conv_norm_group = 4,
        attn_nhead = 8,
        use_mmd=vae_cfg.use_mmd
    ).to(device)
    initialize_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr = vae_cfg.lr, betas = vae_cfg.betas, weight_decay = vae_cfg.weight_decay)
    if vae_cfg.pretrained_path!="":
        load_checkpoints(model,vae_cfg.pretrained_path,optimizer=None,strict=True)
    if parallel: model = nn.DataParallel(model) 

    lr_lambda = CosineWarmupLambda(
        one_epoch_step=len(train_loader),
        warm_up_epochs=100,
        cosine_epochs=50,
        decay_milestones=(200, 400, 600),
        whole_ratios=(0.8, 0.6, 0.4),
        lr_min_ratio=0.1,
    )
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    #create logger
    log_name = "logs" + "_" + time.strftime("%Y%m%d") + ".txt" 
    logger = simple_logger(os.path.join(log_dir, log_name))
    
    # start
    for epoch in range(vae_cfg.epochs):
        epoch = epoch + 1
        # train
        model.train()
        steps_per_epoch = len(train_loader)
        progress_bar = tqdm(total=steps_per_epoch,ncols=200, ascii=True, dynamic_ncols=False)
        progress_bar.set_description(f"train epoch {epoch}")
        accu_loss = torch.zeros((4))  
        for iter, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data_to_device(data,device)
            face_points = data["face_points"]
            face_points_mask =  data["face_points_mask"].float()
            face_mask = data["face_mask"].bool()
            ef_adj = data["ef_adj"]

            points_loss_f, points_mask_loss_f, kl_loss_f,mmd_loss_f = model(face_points, 
                                                                            face_mask, 
                                                                            face_points_mask, 
                                                                            ef_adj,
                                                                            train=True) 
            points_loss_f,points_mask_loss_f = points_loss_f.mean(),points_mask_loss_f.mean()
            if kl_loss_f.ndimension()>0: kl_loss_f,mmd_loss_f = kl_loss_f.mean(),mmd_loss_f.mean()
            total_loss = points_loss_f*vae_cfg.point_loss_weight +  points_mask_loss_f*vae_cfg.point_mask_loss_weight +\
                         kl_loss_f*vae_cfg.kl_loss_weight + mmd_loss_f*vae_cfg.mmd_loss_weight
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=50.0)  
            optimizer.step()
            lr_scheduler.step()

            accu_loss += torch.tensor([points_loss_f.detach(),points_mask_loss_f.detach(),kl_loss_f.detach(),mmd_loss_f.detach()])

            # progress_bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "f": f"{accu_loss[0].item() / (iter + 1):.7f}",
                "fm": f"{accu_loss[1].item() / (iter + 1):.7f}",
                "kl": f"{accu_loss[2].item() / (iter + 1):.5f}",
                "mmd": f"{accu_loss[3].item() / (iter + 1):.5f}",
            })

        #log
        if epoch %vae_cfg.log_per_epoch == 0:
            logger.log(epoch, accu_loss/(iter+1))

        # save
        if epoch % vae_cfg.save_per_epoch == 0:
            print("saved:{}".format(epoch))
            filename = data_name +"_" + "vae" + "_" +"epoch_{:05d}.pth".format(epoch)
            save_checkpoints(model, ckpt_dir, optimizer=None, filename=filename, parallel=parallel)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2,3  python ./train_vae_geom.py --cfg_path ./config/vae_geom_deepcad_f0_30.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2  python ./train_vae_geom.py --cfg_path ./config/vae_geom_abc_f0_50.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2  python ./train_vae_geom.py --cfg_path ./config/vae_geom_furniture_f0_50.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='./config/vae_geom_deepcad_f0_30.yaml')
    args = parser.parse_args()
    main(args)
