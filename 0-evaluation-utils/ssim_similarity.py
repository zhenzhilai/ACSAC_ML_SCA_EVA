# import pytorch_ssim
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from PIL import Image
import sys
import pytorch_ssim


random.seed(0)


reconstructed_folder = sys.argv[1]
print("Read from", reconstructed_folder)
target_folder = sys.argv[2]
TEST_FILES = int(sys.argv[3])
transform = transforms.Compose([
                       transforms.Resize(128),
                       transforms.CenterCrop(128),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])


def compute_ssim(image1, image2):
    img_r = torch.from_numpy(np.rollaxis(cv2.imread(image1), 2)).float().unsqueeze(0)
    img_t = torch.from_numpy(np.rollaxis(cv2.imread(image2), 2)).float().unsqueeze(0)

    if torch.cuda.is_available():
        ssim_value = pytorch_ssim.ssim(img_r.cuda(), img_t.cuda()).detach().cpu().item()
    else:
        ssim_value = pytorch_ssim.ssim(img_r, img_t).detach().cpu().item()
    return ssim_value

def random_select(name_list, N_RANGE, target_file):
    selected_list = random.sample(name_list, N_RANGE)

    # check if target_file is in the selected_list: add it if not
    if target_file not in selected_list:
        selected_list[random.randint(0, N_RANGE-1)] = target_file

    return selected_list


def test(N_RANGE):
    random.seed(0)
    print(N_RANGE)
    name_list = sorted(os.listdir(reconstructed_folder))
    name_list = [name for name in name_list if not name.startswith(".")]

    # if (len(name_list) > TEST_FILES):
    #     name_list = name_list[:TEST_FILES]

    success = 0
    tbar = tqdm(range(TEST_FILES), ncols=100)
    for i in range(TEST_FILES):
        reconstructed_file = reconstructed_folder + '/' + name_list[i]
        
        target_file_list = random_select(name_list, N_RANGE, name_list[i])
        ssim_value_list = []
        for target_file in target_file_list:
            target_file = target_folder + '/' + target_file

            ssim_value = compute_ssim(reconstructed_file, target_file)
            
            ssim_value_list.append(ssim_value)

        max_index = np.argmax(ssim_value_list)
        if target_file_list[max_index] == name_list[i]:
            success += 1
        tbar.set_description("Success: {}".format(success/(i+1)))
        tbar.update(1)
    tbar.close()

    print("Success rate: ", success/TEST_FILES, success)


def main():
    for N_RANGE in [10, 100, 200, 500]:
        test(N_RANGE)

if __name__ == "__main__":
    main()
