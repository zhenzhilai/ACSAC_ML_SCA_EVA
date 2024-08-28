import numpy as np
import os
from scipy.stats import pearsonr
from tqdm import tqdm
import random
import sys

reconstructed_folder = sys.argv[1]
print("Read reconstructed from", reconstructed_folder)

target_folder = sys.argv[2]
print("Read target from", reconstructed_folder)

TEST_FILES = int(sys.argv[3])

random.seed(0)

def read_npz(file):
    npz = np.load(file)
    return npz['arr_0']

def random_select(name_list, N_RANGE, target_file):
    selected_list = random.sample(name_list, N_RANGE)

    # check if target_file is in the selected_list: add it if not
    if target_file not in selected_list:
        selected_list[random.randint(0, N_RANGE-1)] = target_file

    return selected_list

def Compute_PC(n_range):
    random.seed(0)

    # get list of files
    recons_files = sorted(os.listdir(reconstructed_folder))
    recons_files = recons_files[:TEST_FILES]
    target_files = sorted(os.listdir(target_folder))
    target_files = target_files[:TEST_FILES]


    # K=1, N = 10
    success = 0
    tbar = tqdm(range(TEST_FILES), ncols=100)

    for i in range(TEST_FILES):
        recons_file = recons_files[i]
        recons = read_npz(reconstructed_folder + '/' + recons_file)
        pearson_array = []
        target_file_list = random_select(target_files, n_range, recons_file)
        for target in target_file_list:
            target = read_npz(target_folder + '/' + target)
            pearson = pearsonr(recons.flatten(), target.flatten())
            pearson_array.append(abs(pearson[0]))   

        max_index = np.argmax(pearson_array)
        # print("Max Pearson Index: ", max_index)
        if target_file_list[max_index] == recons_file:
            success += 1
        # tbar.set_description("File {}: Success Rate: {:0.4f}".format( target_files[i], success/(i+1)))
        tbar.set_description("Success: {}".format(success/(i+1)))
        tbar.update(1)



    tbar.close()
    print("Overall Success Rate: ", success/TEST_FILES, success)

    return 1

def main():
    for n_range in [10, 100, 200, 500]:
        Compute_PC(n_range)

if __name__ == '__main__':
    main()
