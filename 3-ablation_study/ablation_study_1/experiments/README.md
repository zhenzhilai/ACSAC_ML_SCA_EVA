# Ablation Study 1

## Reconstruct the images from pre-trained models
Download pre-trained models 'ablation1_pretrain.tar' and data 'Celeba_image.tar', 'WB-NUM-data.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it to corresponding directory.
```bash
mkdir model
tar -xvf ablation1_pretrain.tar -C ./model
tar -xvf Celeba_image.tar -C ./data
tar -xvf WB-NUM-data.tar -C ./data
```

- Reconstruct image using R_DTF_DID
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DID-DTF' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DID-DTF" --dim=2 --GAN="FULL" --attn=1 --load_ckpt=../model/R_DTF_DID.pth
```

- Reconstruct image using R_DTF
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DTF' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DTF" --dim=2 --GAN="D" --attn=1 --load_ckpt=../model/R_DTF.pth
```

- Reconstruct image using R_DID
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DID' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DID" --dim=2 --GAN="C" --attn=1 --load_ckpt=../model/R_DID.pth
```

- Reconstruct image using R
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R' --subdir='WB-NUM-data' --trace_lens=384  --notes="R" --dim=2 --GAN="R" --attn=1 --load_ckpt=../model/R.pth
```

## Re-train the model from scratch
- R-DID-DTF
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DID-DTF' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DID-DTF" --dim=2 --GAN="FULL" --attn=1
```

- R-TF
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DTF' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DTF" --dim=2 --GAN="D" --attn=1
```

- R-DID
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R-DID' --subdir='WB-NUM-data' --trace_lens=384  --notes="R-DID" --dim=2 --GAN="C" --attn=1
```

- R
```bash
cd code
python3 main.py --batch_size=100 --lr=2e-4 --num_workers=5 --experiment_name='R' --subdir='WB-NUM-data' --trace_lens=384 --notes="R" --dim=2 --attn=1
```