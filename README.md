# TG-Diff: Coupling Discrete Topology Diffusion and Topology-conditioned Geometry Diffusions for B-Rep Generation

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2607.21928)

![alt TG-Diff](teaser/teaser.png)

TG-Diff is a lightweight two-stage diffusion framework for B-rep generation that decouples topology and geometry modeling. It adopts a surface-centric representation to generate high-quality watertight CAD B-reps efficiently.

## Requirements

### Environment (Tested)
- Linux
- Python 3.10
- CUDA 12.4 
- PyTorch 2.5 

### Dependencies

Install PyTorch and other dependencies:
```
conda create --name tgdiff python=3.10 -y
conda activate tgdiff
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge pythonocc-core==7.5.1
pip install torch_geometric
pip install -r requirements.txt
pip install chamferdist
```

Install OCCWL following the instruction [here](https://github.com/AutodeskAILab/occwl).
If there are missing packages, simply pip install them yourself.






