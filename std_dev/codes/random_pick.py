from cmath import pi
from enum import unique
from json import load
from unittest import loader
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from tqdm import tqdm
import os
import time
from pathlib import Path
import argparse
import random


#def create_random_picks(sourcepath,num_iteration,num_in_subset)
#path = "../allfilter_black/featurestxt"
def create_random_picks(sourcepath,num_subject,num_images):

    num_subject = int(num_subject)
    num_images = int(num_images)

    loaded_txt = np.sort(np.loadtxt(sourcepath,dtype=str))
    
    # initializing empty subject array
    subject_array = np.zeros(shape=(len(loaded_txt),2)).astype(str)

    # create subject_id to index mapping
    for idx,item in enumerate(loaded_txt):
        # Getting the subject id here - be careful here and it depends on your dataset labeling
        subject_id = item.split("/")[-1][:9]
        subject_array[idx][0] = subject_id
        subject_array[idx][1] = idx
    
    total_img_possible = 0

    while total_img_possible < num_images:
        print("The number of total_possible image now is {} and required_image_num is {}. We keep shuffling the subject array".format(total_img_possible,num_images))

        np.random.shuffle(subject_array)
        unique_subject_id = np.unique(subject_array[:,0])
        np.random.shuffle(unique_subject_id)
        balanced_list = []
        
        #picking random subject as first step 
        for counter in np.arange(num_subject):

            index_match = np.where(subject_array[:,0] == unique_subject_id[counter])
            # This is counting the total number of image of the given subject. So, that we can see for this selected subject list, we have enough images as required.
            total_img_possible += index_match[0].size
            matched_items = subject_array[index_match]

            idx = int(matched_items[0][1])
            remove_idx = index_match[0][0]
            subject_array = np.delete(subject_array, remove_idx, axis=0)
            balanced_list.append(loaded_txt[idx])
        
    print("Lenght after one pass which is equal to required num subject:", len(balanced_list))
    print("The total number of images possible is {} and more than num_images so we move on!".format(total_img_possible))
        
    while len(balanced_list) < num_images:
        for counter in np.arange(num_subject):

            index_match = np.where(subject_array[:,0] == unique_subject_id[counter])
            matched_items = subject_array[index_match]

            max_idx = len(matched_items)

            if max_idx > 0 : 

                size = np.random.randint(max_idx)
                idx_to_pick = np.random.choice(max_idx, size=size,replace=False)
        
                if len(idx_to_pick) > 0:
                    idx = matched_items[idx_to_pick][:]
                    mult_retrieve_idx = idx[:,1].astype(int)
                    for idxs in mult_retrieve_idx:
                        if len(balanced_list) < num_images:
                            balanced_list.append(loaded_txt[idxs])
                        else:
                            break
                    remove_idx = index_match[0][idx_to_pick]
                    subject_array = np.delete(subject_array, remove_idx, axis=0)
        

    print("The length of balanced list is:",len(balanced_list))
    
    return balanced_list
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the path of source folder with txt_files and get number of images and subject ID")
    parser.add_argument("--source", "-s", help="source txt files")
    parser.add_argument("--destination", "-d", help="path to save balanced txtfiles")
    parser.add_argument("--num_subject", "-sub", help="number of subject required in balanced list")
    parser.add_argument("--num_images", "-img", help="number of images required in balanced list")
    parser.add_argument("--num_iteration", "-itr", help="total number of iterations")
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    
    
    for iterations in tqdm(np.arange(int(args.num_iteration))):
        balanced_list = create_random_picks(args.source,args.num_subject,args.num_images)
        save_name = os.path.join(args.destination,str(iterations))
        np.savetxt("{}.txt".format(save_name),balanced_list,fmt='%s',newline='\n')
