import numpy as np
import os
from pathlib import Path
import argparse
import re
from tqdm import tqdm


def maskpath_to_imgpath(sourcepath,sourceimgpath):
    files = np.loadtxt(sourcepath,dtype=str)
    img_paths = []
    pattern = re.compile(re.escape("_mask.png"), re.IGNORECASE)
    for items in tqdm(files):
        
        items = pattern.sub(".jpg", str(items))

        img_name = Path(items).name
        img_path = os.path.join(sourceimgpath,img_name)

        img_paths.append(img_path)
            
    return img_paths
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the path of source txt file with maskpath and save the paths to images instead")
    parser.add_argument("--source", "-s", help="path to txt with all the maskpaths")
    parser.add_argument("--imgpath", "-img", help="path to the directory where images are stored")
    parser.add_argument("--dest", "-d", help="Destination to save the file")
    parser.add_argument("--name", "-n", help="Name to save the txt file")
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # Current implementation only supports if there are no subdirectories - but a tiny edit can fix this.
    save_list = maskpath_to_imgpath(args.source,args.imgpath)
    save_name = os.path.join(args.dest,args.name)

    np.savetxt("{}.txt".format(save_name),save_list,fmt="%s",newline='\n')    
    