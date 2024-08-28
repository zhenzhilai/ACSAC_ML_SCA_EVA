# Instructions for running different evaluation scripts

## Face++ matching rate
### Query Face++ API
We use script ```face_similarity.py``` to compute the face-matching rate of a reconstructed image and its reference image. 
The results are stored in a file.
You may need to register an account at https://www.faceplusplus.com.

If you have a new API key, you need to first configure the API key and secret in ```face_similarity.py```. Then, run:
```bash
python3 face_similarity.py [input_folder_path] [output_face++_result_csv_file] [number_of_tested_images] [reference_folder_path] [if_print]
```
- **input_folder_path:** the folder path storing the reconstructed images.
- **output_face++_result_csv_file:** the output file to store the face++ result.
- **number_of_tested_images:** number. The number of images you want to test.
- **reference_folder_path:** the folder path storing the reference images.
- **if_print:** 0 or 1. 1 for print the results when num reaches 1000.

### Calculate matching rate
We use script ```analyze_face++.py``` to read from a file contains the results returned by the Face++ API. We analyze the results with three error rate thresholds and a fixed threshold of 50. Please refer to line 49 of the script for detailed information. 

```bash
python3 analyze_face++.py [face++_result_csv_file]
```
- **face++_result_csv_file:** the csv file storing the face++ result.

## Distinguishability Evaluation with SSIM
We use script ```ssim_similarity.py``` to perform the distinguishability evaluation. In this script, we use SSIM to measure the similarity between the reconstructed result and its reference input.
```bash
python3 ssim_similarity.py [input_folder_path] [reference_folder_path] [number_of_tested_images] > [output_result_file]
```
- **input_folder_path:** the folder path storing the reconstructed images.
- **reference_folder_path:** the folder path storing the reference images.
- **number_of_tested_images:** number. The number of images you want to test.
- **output_result_file:** the file storing the distinguishability result


## Distinguishability Evaluation with Pearson Correlation
We use script ```Pearson_similarity.py``` to perform the distinguishability evaluation. In this script, we use Pearson correlation to measure the similarity between the reconstructed result and its reference input.
```bash
python3 Pearson_similarity.py [input_folder_path] [reference_folder_path] [number_of_tested_images] > [output_result_file]
```
- **input_folder_path:** the folder path storing the reconstructed images.
- **reference_folder_path:** the folder path storing the reference images.
- **number_of_tested_images:** number. The number of images you want to test.
- **output_result_file:** the file storing the distinguishability result