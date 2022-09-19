import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm

#path = "../allfilter_black/featurestxt"
def create_featurestxt(sourcepath,featurespath,destinationpath):
    files = os.listdir(sourcepath)
    for items in tqdm(files):
        cat_name = Path(items).stem
        loaded_txt = np.loadtxt(os.path.join(sourcepath,items),dtype=str)
        path_featurestxt = []
        for item in loaded_txt:
            img_name = Path(item).name.replace(".JPG",".npy")
            folder_name = Path(item).parent.name

            path_feature = os.path.join(os.path.abspath(featurespath),folder_name,img_name)
            #print(path_feature)
            if os.path.exists(path_feature):
                path_featurestxt.append(path_feature)
            else:
                print("path not found",item)
        
        save_name = os.path.join(destinationpath,cat_name)
        #print("Filed saved at {} as {}.txt".format(destinationpath,cat_name))

        np.savetxt("{}.txt".format(save_name),path_featurestxt,fmt='%s',newline='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the path of source folder with txt_files and get number of images and subject ID")
    parser.add_argument("--source", "-s", help="loadpath to all txtfiles")
    parser.add_argument("--features", "-f", help="Location of folder where features are stored")
    parser.add_argument("--destination", "-d", help="path to save modified txtfiles")
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    
    create_featurestxt(args.source,args.features,args.destination)
    
