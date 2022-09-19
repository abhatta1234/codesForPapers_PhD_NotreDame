import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def parse_bisenet_hair_ratio(path):
    hair_ratio_list = []
    for subdir, dirs, files in os.walk(path):
        print(subdir)
        for filename in tqdm(files):
            #For BiSeNet segmentation there is a color mask version and mask version. Just using the mask!
            if filename.split("_")[-2] != "color":
                img = cv2.imread(subdir + os.sep + filename, cv2.IMREAD_GRAYSCALE)

                # 17 is the hair region
                img[img!=17] = 0
                img[img== 17] = 1

                hair_area = len(img[img == 1])
                total_area = img.shape[0] * img.shape[1]

                per_skin = round(hair_area/total_area *100,2)

                filename = filename.replace("_mask.png", ".JPG")

                hair_ratio_list.append([filename,per_skin])

    return hair_ratio_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the hair ratio for all images given the path to BiSeNet segmentation mask")
    parser.add_argument("--source", "-s", required = True, help="Path to source directory of BiseNet. Make sure this is for each group - each separate for C_M,C_F and so on")
    parser.add_argument("--destination", "-d", required = True, help="path to save txt file")
    parser.add_argument("--name", "-n", required = True, help="Name to save the file.")
    args = parser.parse_args()

    hair_ratio_list = parse_bisenet_hair_ratio(args.source)

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    
    save_name = os.path.join(args.destination,args.name)

    np.savetxt("{}.txt".format(save_name), hair_ratio_list, fmt='%s', delimiter=' ', newline='\n')