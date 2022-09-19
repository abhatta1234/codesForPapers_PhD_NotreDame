
from pathlib import Path
import os
from tqdm import tqdm
import time
import argparse
import numpy as np
import re

#Sample paths
sample_pathtxtimage = "iou_exp/hair_color/txt_files/AA_M_blackhair_img_beardfilter.txt"
sample_pathtxtmask = "AA_M_masklist.txt"
sample_maskdirectory = "/afs/crc.nd.edu/user/a/abhatta/Shared/MORPH_Segmentation/AA_M/"
sample_destination = "iou_exp/hair_color/txt_files/"

def save_existing_mask(img_path,mask_path,crc_mask_dirpath):
    img_loaded = np.loadtxt(img_path,dtype=str)
    print("Number of Images:",len(img_loaded))
    mask_loaded = np.loadtxt(mask_path,dtype=str)
    print("Total Number of mask to search from : ",len(mask_loaded))
    final_list = []

    #check if the directories has multiple subdirectories:
    num_subdirectory = len(next(os.walk(crc_mask_dirpath))[1])
    print("The numbner of Sub-Directory: ",num_subdirectory)

    pattern = re.compile(re.escape(".JPG"), re.IGNORECASE)
    for items in tqdm(img_loaded):
        path = Path(items)

        if num_subdirectory:
            name = os.path.join(path.parent.name,path.name)
        else:
            name = path.name

        #Substitution .jpg/.JPG with _mask.png in case insensitive fashion
        name = pattern.sub("_mask.png", str(name))
        crc_name = os.path.join(crc_mask_dirpath,name)
        
        
        if crc_name in mask_loaded and os.path.exists(crc_name):
            final_list.append(crc_name) 
        else:
            print("No Mask Found")
    return final_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Given a list of imgpath, checks if mask for that image is available or not and create a list of all maskpaths that are available!")
    parser.add_argument("--pathtxtimage","-img", help="path to txtfile with list of images")
    parser.add_argument("--pathtxtmask","-mask",help="path to txtfile with list of masks")
    parser.add_argument("--maskdirectory","-maskdir",help="mask directory ")
    parser.add_argument("-destination",'-d',help="path to save the returned mask path")
    parser.add_argument("-name",'-n',help="name to save txt file")

    args = parser.parse_args()

    save_list = save_existing_mask(img_path=args.pathtxtimage,mask_path=args.pathtxtmask,crc_mask_dirpath=args.maskdirectory)
    print("Number of Masks Found:",len(save_list))

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    save_path = args.destination+args.name+"_nobeard_notbald_maskpath.txt"
    np.savetxt(save_path,save_list,fmt="%s",newline='\n')
    
