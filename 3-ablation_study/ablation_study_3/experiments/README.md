# Ablation Study 3

## Reconstruct the images from pre-trained models
Download pre-trained models 'ablation3_pretrain.tar' and data 'Celeba_image.tar', 'BB-NUM-data.tar', 'BB-OH-data.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it to corresponding directory.
```bash
mkdir model
tar -xvf ablation3_pretrain.tar -C ./model
tar -xvf Celeba_image.tar -C ./data
tar -xvf BB-NUM-data.tar -C ./data
tar -xvf BB-OH-data.tar -C ./data
```

- AE+D_NUM:
```bash
python3 code_num/recons_image_nogan.py --exp_name=AE+D_NUM --batch_size=50 --num_workers=5 --load_ckpt=model/AE+D_NUM.pth
```

- AE_NUM:
```bash
python3 code_num/recons_image.py --exp_name=AE_NUM --batch_size=50 --num_workers=5 --load_ckpt=model/AE_NUM.pth
```

- AE+D_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=50. Please reduce the size depending on your GPU capacity.**
```bash
python3 code_oh/my_train.py --exp_name=AE+D_OH --batch_size=50 --num_workers=5 --load_ckpt=model/AE+D_OH.pth
```

- AE_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=50. Please reduce the size depending on your GPU capacity.**
```bash
python3 code_oh/my_train_nogan.py --exp_name=AE_OH --batch_size=50 --num_workers=5 --load_ckpt=model/AE_OH.pth
```

## Re-train the model from scratch
- AE+D_NUM:
```bash
python3 code_num/recons_image_nogan.py --exp_name=AE+D_NUM --batch_size=50 --num_workers=5
```

- AE_NUM:
```bash
python3 code_num/recons_image.py --exp_name=AE_NUM --batch_size=50 --num_workers=5
```

- AE+D_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=50. Please reduce the size depending on your GPU capacity.**
```bash
python3 code_oh/my_train.py --exp_name=AE+D_OH --batch_size=50 --num_workers=5
```

- AE_OH:

**Warning: this task requires a huge GPU-RAM with batch_size=50. Please reduce the size depending on your GPU capacity.**
```bash
python3 code_oh/my_train_nogan.py --exp_name=AE_OH --batch_size=50 --num_workers=5
```