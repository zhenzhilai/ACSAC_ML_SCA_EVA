# Reproduce Manifold-SCA

## Evaluation results are ready in 'results'
- Instrumented Images (refiner-off)
```bash
python3 ../../0-evaluation-utils/analyze_face++.py results/recons_CelebA_refiner_off_face++.csv
```
- Practical PP Images (refiner-off)
```bash
python3 ../../0-evaluation-utils/analyze_face++.py results/CelebA_intel_dcache_refiner_off_face++.csv
```

## If you want to recompute the results
Download our images 'reproduce_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf reproduce_images.tar
```

- Instrumented Images (refiner-off)
```bash
python3 ../../0-evaluation-utils/face_similarity.py recons_CelebA_refiner_off/recons ins_off.csv 1000 reference_test 1
```

```bash
python3 ../../0-evaluation-utils/analyze_face++.py ins_off.csv
```

- Practical PP Images (refiner-off)
```bash
python3 ../../0-evaluation-utils/face_similarity.py CelebA_intel_dcache_refiner_off/recons/test pp_off.csv 1000 reference_test 1
```

```bash
python3 ../../0-evaluation-utils/analyze_face++.py pp_off.csv
```
