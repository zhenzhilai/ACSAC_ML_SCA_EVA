# Ablation Study 1

## Evaluation results are ready in 'results'

### SIM1: Average SSIM
Read from the test during the training process directly. Please refer to "../experiment" to get the data from loading the pre-trained model.

### SIM2: Face++
- R_DTF_DID
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/R_DTF_DID_face++.csv
```
- R_DTF
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/R_DTF_face++.csv
```
- R_DID
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/R_DID_face++.csv
```
- R
```bash
python3 ../../../0-evaluation-utils/analyze_face++.py results/R_face++.csv
```

### DIS: Distinguishability
- R_DTF_DID
```bash
tail -f results/R_DTF_DID.out
```
- R_DTF
```bash
tail -f results/R_DTF.out
```
- R_DID
```bash
tail -f results/R_DID.out
```
- R
```bash
tail -f results/R.out
```


## If you want to recompute the results
Download our images 'ablation1_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf ablation1_images.tar
```

### Face++
- R_DTF_DID
```bash
python3 ../../../0-evaluation-utils/face_similarity.py R_DTF_DID/test R_DTF_DID_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py R_DTF_DID_face++.csv
```
- R_DTF
```bash
python3 ../../../0-evaluation-utils/face_similarity.py R_DTF/test R_DTF_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py R_DTF_face++.csv
```

- R_DID
```bash
python3 ../../../0-evaluation-utils/face_similarity.py R_DID/test R_DID_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py R_DID_face++.csv
```

- R
```bash
python3 ../../../0-evaluation-utils/face_similarity.py R/test R_face++.csv 10000 reference_test 1
python3 ../../../0-evaluation-utils/analyze_face++.py R_face++.csv
```

### Distinguishability
- R_DTF_DID
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py R_DTF_DID/test reference_test 10000 > R_DTF_DID.out
tail -f R_DTF_DID.out
```
- R_DTF
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py R_DTF/test reference_test 10000 > R_DTF.out
tail -f R_DTF.out
```

- R_DID
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py R_DID/test reference_test 10000 > R_DID.out
tail -f R_DID.out
```

- R
```bash
python3 ../../../0-evaluation-utils/ssim_similarity.py R/test reference_test 10000 > R.out
tail -f R.out
```
