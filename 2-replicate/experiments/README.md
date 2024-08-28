# Replicate Manifold-SCA

## Reconstruct the images from pre-trained models
Download pre-trained models 'replicate_pretrain.tar' and data 'Celeba_image.tar', 'manifold_cacheline.tar', 'manifold_l1dpp_cacheline.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it to corresponding directory.
```bash
mkdir model
tar -xvf replicate_pretrain.tar -C ./model
tar -xvf Celeba_image.tar -C ./data
tar -xvf manifold_cacheline.tar -C ./data
tar -xvf manifold_l1dpp_cacheline.tar -C ./data
```

- Reconstruct image from instrumented activities
```bash
python3 replicate_manifold_intel_pin_code/recons_image_pretrain.py --pretrain_path=model/replicate_instrumented.pth --exp_name=recons_pin
```

- Reconstruct image from practical Prime-Probe activities
```bash
python3 replicate_manifold_intel_pp_code/recons_image_pretrain.py --pretrain_path=model/replicate_practical.pth --exp_name=recons_pp
```

## Re-train the model from scratch
- Reconstruct image from instrumented activities
```bash
python3 replicate_manifold_intel_pin_code/recons_image.py --exp_name=test_pin
```

- Reconstruct image from practical Prime-Probe activities
```bash
python3 replicate_manifold_intel_pp_code/recons_image.py --exp_name=test_pp
```