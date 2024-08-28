# Ablation Study 2

## Reconstruct the images from pre-trained models
Download pre-trained models 'ablation2_pretrain.tar' and data 'Celeba_image.tar', 'WB-NUM-data.tar', 'WB-OH-data.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it to corresponding directory.
```bash
mkdir model
tar -xvf ablation2_pretrain.tar -C ./model
tar -xvf Celeba_image.tar -C ./data
tar -xvf WB-NUM-data.tar -C ./data
tar -xvf WB-OH-data.tar -C ./data
```

- Reconstruct image using 2D_NUM:
Please refer to Ablation Study 1: **R**

- Reconstruct image using 1D_NUM:
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='num-1D' --subdir='WB-NUM-data' --trace_lens=384  --notes="1d-num" --dim=1 --attn=1 --load_ckpt=../model/1D_NUM.pth
```

- Reconstruct image using 1D_OH:
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='onehot-1D' --subdir='WB-OH-data' --trace_lens=24576  --notes="1d-oh" --dim=1 --attn=1 --load_ckpt=../model/1D_OH.pth
```

- Reconstruct image using 2D_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=100. Please reduce the size depending on your GPU capacity.**
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='onehot2d-2d' --subdir='WB-OH-data' --trace_lens=24576 --notes="2d-oh" --dim=2 --attn=1 --load_ckpt=../model/2D_OH.pth
```

## Re-train the model from scratch
- 2D_NUM:
Please refer to Ablation Study 1: **R**

- 1D_NUM:
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='num-1D' --subdir='WB-NUM-data' --trace_lens=384  --notes="1d-num" --dim=1 --attn=1 
```

- 1D_OH:
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='onehot-1D' --subdir='WB-OH-data' --trace_lens=24576  --notes="1d-oh" --dim=1 --attn=1 
```

- 2D_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=100. Please reduce the size depending on your GPU capacity.**
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='onehot2d-2d' --subdir='WB-OH-data' --trace_lens=24576 --notes="2d-oh" --dim=2 --attn=1 
```