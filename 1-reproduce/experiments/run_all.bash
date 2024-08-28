#!/bin/bash
#### please activate your python environment first
# conda activate cnn

export MANIFOLD_SCA=$(pwd)
cd code

## Reproduce instrumented activities
python output.py --dataset="CelebA" --exp_name="reproduce_intel_pin" --use_refiner=0

## Reproduce practical activities
python pp_image.py --exp_name="reproduc_intel_dcache" --use_refiner=0
