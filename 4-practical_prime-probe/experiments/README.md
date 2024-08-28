# Practical Prime-Probe

## Reconstruct from pre-trained models
Download pre-trained models 'practical_pp_pretrain.tar' and data 'Celeba_image.tar', 'practical_pp_data.tar', 'target_activities_data.tar', 'reconstructed_activities_data.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it to corresponding directory.
```bash
mkdir model
tar -xvf practical_pp_pretrain.tar -C ./model
tar -xvf Celeba_image.tar -C ./data
tar -xvf practical_pp_data.tar -C ./data
tar -xvf target_activities_data.tar -C ./data
tar -xvf reconstructed_activities_data.tar -C ./data
```

### Baseline
Please see 2-replicated: **Reconstruct image from practical Prime-Probe activities**

### EXP 0: Identify cache sets related to pixel-related activities
```bash
cd code_for_identifying_pixel_related_cache_sets
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=5 --experiment_name='pp_identifying_pixel_related_cache_sets' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to pixel related cache sets" --dim=2 --target="idct" --idct_out_dim=1 --attn=1 --load_ckpt=../model/identify_relevant_cache_set.pth
```

### EXP 1: Reconstruct simplified instrumented cache activities
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='pp_activities_to_pixel_related_activities' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to pixel-related activities" --dim=2 --target="idct" --idct_out_dim=384 --attn=1 --load_ckpt=../model/EXP_1.pth
```

### EXP 2 pre-trained: Pre-trained model
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='simplified_ideal_trace_to_img_attn' --subdir='target_activities_data' --trace_lens=384 --notes="simplified instrumented to pixel-related activities to img" --dim=2 --target="img" --attn=1 --load_ckpt=../model/EXP_2_pretrained.pth
```

### EXP 2: Reconstruct images from reconstructed simplified instrumented cache activities(EXP 1)
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='reconstructed_pixel_related_activities_to_img' --subdir='reconstructed_activities_data' --trace_lens=384 --notes="reconstructed trace to img" --dim=2 --target="img" --recon_trace_to_img=1 --attn=1 --load_ckpt=../model/EXP_2_pretrained.pth
```

### EXP 3: Reconstruct images from prime-probe activities directly
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='pp_trace_to_img_directly' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to img directly" --dim=2 --target="img" --attn=1 --load_ckpt=../model/EXP_3.pth
```



## Re-train the model from scratch

### Baseline
Please see 2-replicated: **Reconstruct image from practical Prime-Probe activities**

### EXP 0: Identify cache sets related to pixel-related activities
```bash
cd code_for_identifying_pixel_related_cache_sets
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=5 --experiment_name='pp_identifying_pixel_related_cache_sets' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to pixel related cache sets" --dim=2 --target="idct" --idct_out_dim=1 --attn=1
```

### EXP 1: Reconstruct Simplified Instrumented Cache Activities
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='pp_activities_to_pixel_related_activities' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to pixel-related activities" --dim=2 --target="idct" --idct_out_dim=384 --attn=1
```

### EXP 2 pre-trained: Pre-trained model
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='simplified_ideal_trace_to_img' --subdir='target_activities_data' --trace_lens=384 --notes="simplified instrumented to pixel-related activities to img" --dim=2 --target="img" --attn=1
```

### EXP 2: Reconstruct images from reconstructed simplified instrumented cache activities(EXP 1)
There is no training in this EXP. Please refer to the previous EXP 2.2 to load the pretrained model and reconstructed activities to get the result.

### EXP 3: Reconstruct images from prime-probe activities directly
```bash
cd code
python3 main.py --batch_size=100 --epochs=50 --lr=2e-4 --num_workers=4 --experiment_name='pp_trace_to_img_directly' --subdir='practical_pp_data' --trace_lens=800 --notes="practical activities to img directly" --dim=2 --target="img" --attn=1
```
