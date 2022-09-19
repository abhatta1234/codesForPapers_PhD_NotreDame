import argparse
from pathlib import Path
import os
import numpy as np


dir_path = "/afs/crc.nd.edu/user/a/abhatta/Shared/MORPH_Segmentation/AA_M"

def create_list(directory_path):
    img_list = []
    for subdir, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.split("_")[-2] != "color":
                img_list.append(os.path.join(subdir,filename))
    return img_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a list for mask by crawling directory")
    parser.add_argument("--directory","-dir", help="path of directory to crawl")
    parser.add_argument("--destination","-dest",help="destination to save txt file")
    parser.add_argument("--name","-n",help="name to save txt file")

    args = parser.parse_args()

    save_list = create_list(args.directory)

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
   
    save_name = os.path.join(args.destination,args.name) + "_masklist.txt"

    np.savetxt(save_name,save_list,fmt="%s",newline='\n')


