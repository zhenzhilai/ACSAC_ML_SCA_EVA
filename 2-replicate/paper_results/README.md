# Replicate Manifold-SCA

## Evaluation results are ready in 'results'
### Face++
- Instrumented Images
```bash
python3 ../../0-evaluation-utils/analyze_face++.py results/replicate_instrumented_face++.csv
```
- Practical PP Images
```bash
python3 ../../0-evaluation-utils/analyze_face++.py results/replicate_practical_face++.csv
```

### Distinguishability
- Instrumented Images
```bash
tail -f results/dis_replicate_instrumented_final.out
```
- Practical PP Images
```bash
tail -f results/dis_replicate_practical_final.out
```



## If you want to recompute the results
Download our images 'replicate_images.tar' from [Goolge Drive](https://drive.google.com/drive/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB?usp=sharing). Extract it there.
```bash
tar -xvf replicate_images.tar
```

### Face++
- Instrumented Images
```bash
python3 ../../0-evaluation-utils/face_similarity.py replicate_instrumented ins.csv 1000 reference_test 1
```

```bash
python3 ../../0-evaluation-utils/analyze_face++.py ins.csv
```

- Practical PP Images
```bash
python3 ../../0-evaluation-utils/face_similarity.py replicate_practical pp.csv 1000 reference_test 1
```

```bash
python3 ../../0-evaluation-utils/analyze_face++.py pp.csv
```

### Distinguishability
- Instrumented Images
```bash
python3 ../../0-evaluation-utils/ssim_similarity.py replicate_instrumented reference_test 10000 > ins.out
```

```bash
tail -f ins.out
```

- Practical PP Images
```bash
python3 ../../0-evaluation-utils/ssim_similarity.py replicate_practical reference_test 10000 > pp.out
```

```bash
tail -f pp.out
```