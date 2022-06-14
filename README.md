# Usage
First, clone the repository locally:
```
git clone https://github.com/PengWan-Yang/few-shot-transformer
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
To train the model on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py
```
