# Practical Prime-Probe

## Evaluation results are ready in 'results'

### Accuracy for EXP 0: Identify cache sets related to pixel-related activities
The result is read during the test in training. Please refer to 'experiments' EXP 0 to get the result from loading the pre-trained model

### DIS: Distinguishability

- Baseline
Please see 2-replicated: **Reconstruct image from practical Prime-Probe activities**

- EXP 1: Reconstruct simplified instrumented cache activities
```bash
tail -f results/EXP_1.out
```

### EXP 2 pre-trained: Pre-trained model
```bash
tail -f results/EXP_2_pretrained.out
```

### EXP 2: Reconstruct images from reconstructed simplified instrumented cache activities(EXP 1)
```bash
tail -f results/EXP_2.out
```

### EXP 3: Reconstruct images from prime-probe activities directly
```bash
tail -f results/EXP_3.out
```


## If you want to recompute the results
Download our images 'practical_pp_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf practical_pp_images.tar
```


### Distinguishability
- Baseline
Please see 2-replicated: **Reconstruct image from practical Prime-Probe activities**

- EXP 1: Reconstruct simplified instrumented cache activities
```bash
python3 ../../0-evaluation-utils/Pearson_similarity.py EXP_1/test reference_test_activities 10000 > EXP_1.out
tail -f EXP_1.out
```

### EXP 2 pre-trained: Pre-trained model
```bash
python3 ../../0-evaluation-utils/ssim_similarity.py EXP_2_pretrained/test reference_test 10000 > EXP_2_pretrained.out
tail -f EXP_2_pretrained.out
```

### EXP 2: Reconstruct images from reconstructed simplified instrumented cache activities(EXP 1)
```bash
python3 ../../0-evaluation-utils/ssim_similarity.py EXP_2/test reference_test 10000 > EXP_2.out
tail -f EXP_2.out
```

### EXP 3: Reconstruct images from prime-probe activities directly
```bash
python3 ../../0-evaluation-utils/ssim_similarity.py EXP_3/test reference_test 10000 > EXP_3.out
tail -f EXP_3.out
```

