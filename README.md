![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
# Artifact of Paper: *R+R: Demystifying ML-Assisted Side-Channel Analysis Framework: A Case of Image Reconstruction*

The paper has been accepted by ACSAC 2024.  
In this repo, we provide experiment results of this paper, evaluation scripts and code to facilitate the reproducibility and the replicability of our work.

We provide an introduction of each experiment in the Artifact Appendix.

The [Google drive](https://drive.google.com/drive/u/1/folders/1oyqViKeu3LpqDGozCDVpA70OewqAJQSB) contains all the necessary data to replicate our experiments. More details will be explained in each Folders's.

## Folders
For verifying, reproducing, and replicating our results, please go to '1-reproduce', '2-replicate', '3-ablation_study/ablation_study_x' and '4-practical_prime-probe' and follow:
- Verify: please go to sub-directory '/paper_results' (i.e., 1-reproduce/paper_results) and follow the instructions in its README.md.
- Reproduce: please go to sub-directory '/experiments' (i.e. 1-reproduce/paper_results) and follow the instructions in its README.md 'Reconstruct the images from pre-trained models'. Note that it only reconstruct images or cache activities. To compute the evaluation results, you need to use similar commands stated in '/paper_results' and modify the 'input_folder_path' to the newly reconstructed images or cache activities.
- Replicate: please go to sub-directory '/experiments' (i.e. 1-reproduce/paper_results) and follow the instructions in its README.md 'Re-train the model from scratch'. Note that it is a simple training and reconstruction. To compute the evaluation results, you need to use similar commands stated in '/paper_results' and modify the 'input_folder_path' to the newly reconstructed images or cache activities.

### 0-evaluation-utils
We provide evaluation scripts that analyze the face-matching rate and the distinguishability evaluation. These scripts will be invoked by following folders to compute the benchmarks.

### 1-reproduce
We provide scripts to evaluate the results of Manifold-SCA. 

### 2-replicate
We provide scripts and code to replicate the results of the Manifold-SCA framework.

### 3-ablation_study
We provide scripts and code to perform the ablation study of the Manifold-SCA framework.

### 4-practical_prime-probe
We provide scripts and code to analyze practical cache activities.


## Environment Settings
Please first activate your Python environment. Then install all dependence by:
```bash
pip install -r requirements.txt
```

### Possible Variance:
Note that there will be some varience in the benchmark results if you compute it by loading the pretrained model to re-generate images. For example, the float-point computation in different GPU may not always be the same, leading to a different round-up in some results. Also, the batch size matters. However, you are still expected to observe similar results.
