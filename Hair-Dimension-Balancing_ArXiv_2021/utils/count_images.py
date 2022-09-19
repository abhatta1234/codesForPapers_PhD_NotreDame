import numpy as np
import os
from pathlib import Path
import argparse

#path = "../allfilter_black/featurestxt"
def check_unique(sourcepath):
    files = os.listdir(sourcepath)
    
    for items in files:
        if items.endswith(".txt"):
            cat_name = Path(items).stem
            loaded_txt = np.loadtxt(os.path.join(sourcepath,items),dtype=str)
            subject_id = []
            for item in loaded_txt:
                #print(item,item.split("/")[-1][:9])
                subject_id.append(item.split("/")[-1][:9])
                
            subject_id = np.asarray(subject_id)
            print("The file name is {}, total item is {}, total_unique_items is {} and total unique subjects is {}".format(cat_name,len(loaded_txt),len(np.unique(loaded_txt)),len(np.unique(subject_id))))
            print("\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the path of source folder with txt_files and get number of images and subject ID")
    parser.add_argument("--source", "-s", help="hair attribute path")
    args = parser.parse_args()
    check_unique(args.source)

    

