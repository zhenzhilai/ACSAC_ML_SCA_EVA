# Ablation Study 3

## Evaluation results are ready in 'results'

### SIM1: Average SSIM
Read from the test during the training process directly. Please refer to "../experiment" to get the data from loading the pre-trained model.

### SIM2: Face++
- AE+D_NUM:
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/AE+D_NUM_face++.csv
```

- AE_NUM:
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/AE_NUM_face++.csv
```

- AE+D_OH:
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/AE+D_OH_face++.csv
```

- AE_OH:
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/AE_OH_face++.csv
```

### DIS: Distinguishability
- AE+D_NUM:
```bash
tail -f results/AE+D_NUM.out
```

- AE_NUM:
```bash
tail -f results/AE_NUM.out
```

- AE+D_OH:
```bash
tail -f results/AE+D_OH.out
```

- AE_OH:
```bash
tail -f results/AE_OH.out
```


## If you want to recompute the results
Download our images 'ablation3_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf ablation3_images.tar
```

### Face++
- AE+D_NUM:
```bash
python3 ../../../0-evaluation-utils/face_similarity.py AE+D_NUM AE+D_NUM_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py AE+D_NUM_face++.csv
```

- AE_NUM:
```bash
python3 ../../../0-evaluation-utils/face_similarity.py AE_NUM AE_NUM_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py AE_NUM_face++.csv
```
- AE+D_OH:
```bash
python3 ../../../0-evaluation-utils/face_similarity.py AE+D_OH AE+D_OH_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py AE+D_OH_face++.csv
```
- AE_OH:
```bash
python3 ../../../0-evaluation-utils/face_similarity.py AE_OH AE_OH_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py AE_OH_face++.csv
```

### Distinguishability
- AE+D_NUM:
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py AE+D_NUM reference_test 10000 > AE+D_NUM.out
tail -f AE+D_NUM.out
```

- AE_NUM:
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py AE_NUM reference_test 10000 > AE_NUM.out
tail -f AE_NUM.out
```

- AE+D_OH:
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py AE+D_OH reference_test 10000 > AE+D_OH.out
tail -f AE+D_OH.out
```

- AE_OH:
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py AE_OH reference_test 10000 > AE_OH.out
tail -f AE_OH.out
```
