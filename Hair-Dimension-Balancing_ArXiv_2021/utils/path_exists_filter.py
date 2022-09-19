import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import argparse

#path = "../allfilter_black/featurestxt"
def path_exists_filter(sourcepath):
    files = os.listdir(sourcepath)
    for items in files:
        cat_name = Path(items).stem
        print(cat_name)
        loaded_txt = np.loadtxt(os.path.join(sourcepath,items),dtype=str)
        filtered_path = []
        for item in tqdm(loaded_txt):
            if os.path.exists(item):
                filtered_path.append(item)
            else:
                print("path not found",item)
        
        save_name = os.path.join(sourcepath,cat_name)
        np.savetxt("{}.txt".format(save_name),filtered_path,fmt='%s',newline='\n')
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the path of source folder with txt_files and get number of images and subject ID")
    parser.add_argument("--source", "-s", help="hair attribute path")
    args = parser.parse_args()
    path_exists_filter(args.source)
