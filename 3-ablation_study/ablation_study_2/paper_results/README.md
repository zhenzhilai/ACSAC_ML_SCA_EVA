# Ablation Study 2

## Evaluation results are ready in 'results'

### SIM1: Average SSIM
Read from the test during the training process directly. Please refer to "../experiment" to get the data from loading the pre-trained model.

### SIM2: Face++
- 2D_NUM
Please refer to Ablation Study 1: **R**
- 1D_NUM
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/1D_NUM_face++.csv
```
- 1D_OH
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/1D_OH_face++.csv
```
- 2D_OH
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/2D_OH_face++.csv
```

### DIS: Distinguishability
- 2D_NUM
Please refer to Ablation Study 1: **R**
- 1D_NUM
```bash
tail -f results/1D_NUM.out
```
- 1D_OH
```bash
tail -f results/1D_OH.out
```
- 2D_OH
```bash
tail -f results/2D_OH.out
```


## If you want to recompute the results
Download our images 'ablation2_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf ablation2_images.tar
```

### Face++
- 2D_NUM
Please refer to Ablation Study 1: **R**

- 1D_NUM
```bash
python3 ../../../0-evaluation-utils/face_similarity.py 1D_NUM/test 1D_NUM_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py 1D_NUM_face++.csv
```
- 1D_OH
```bash
python3 ../../../0-evaluation-utils/face_similarity.py 1D_OH/test 1D_OH_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py 1D_OH_face++.csv
```
- 2D_OH
```bash
python3 ../../../0-evaluation-utils/face_similarity.py 2D_OH/test 2D_OH_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py 2D_OH_face++.csv
```

### Distinguishability
- 2D_NUM
Please refer to Ablation Study 1: **R**

- 1D_NUM
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py 1D_NUM/test reference_test 10000 > 1D_NUM.out
tail -f 1D_NUM.out
```

- 1D_OH
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py 1D_OH/test reference_test 10000 > 1D_OH.out
tail -f 1D_OH.out
```

- 2D_OH
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py 2D_OH/test reference_test 10000 > 2D_OH.out
tail -f 2D_OH.out
```
