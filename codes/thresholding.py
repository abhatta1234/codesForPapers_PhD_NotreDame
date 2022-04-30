import numpy as np
import argparse
import ast # this package to convert dict or list like looking str to actual dict or list
import os
from os import path,makedirs
from sklearn.feature_extraction import img_to_graph
from tqdm import tqdm
from pathlib import Path

img_path = "/afs/crc.nd.edu/user/a/abhatta/MORPH3/"

def get_attribute_values(items,img_path,group):
    # all items in txt file are stored as str, this function helps convert
    # string in dictionary looking format to actual dictionary or list format

    # parsing values here
    msft_facial_hair = ast.literal_eval(items[3])
    msft_bald = ast.literal_eval(items[1])
    amzn_beard = ast.literal_eval(items[4])

    # getting respective values
    msft_beard_val = msft_facial_hair["beard"]
    msft_moustache_val = msft_facial_hair["moustache"]
    msft_sideburn_val = msft_facial_hair["sideburns"]
    amzn_beard_bool = amzn_beard[0]
    amzn_beard_val = amzn_beard[1]
    msft_bald_val = msft_bald["bald"]
    bisenet_ratio = float(items[2])

    #THRESHOLDING for beard

    # This is saying if both agrees if the image has beard or not
    # Thresholding was .4, because at that point beard prediction looked decent
    # below 55 confidence of amazon, beard prediction are too noisy

    msft_pred = msft_beard_val>=0.4 or msft_moustache_val>=0.4 or msft_sideburn_val>=0.4
    amzn_pred = (amzn_beard_bool==True and amzn_beard_val>55) or (amzn_beard_bool==False and amzn_beard_val<65)

    # High prediction for Amazon - if amzn predicts an image with more than 85 confidence for beard, then it has beard regardless of microsoft prediction
    amzn_high_confd = (amzn_beard_bool==True and amzn_beard_val>85)
    # High prediction for microsoft
    # if any of beard,moustache or sideburn value are more than 0.6, then image has facial hair regardless of amzn prediction
    msft_high_cofd = (msft_beard_val>=0.6 or msft_moustache_val>=0.6 or msft_sideburn_val>=0.6) 
        
    # THRESHOLDING for baldness

    # low_hair_ratio from BiSeNet
    low_hair_ratio = bisenet_ratio < 1.5
    # high bald prediction
    high_bald_microsoft = msft_bald_val>=.97
    # for fuzzy data, if both agress with high confidencea
    combined_result = bisenet_ratio < 2 and msft_bald_val>=.90

    # getting the respective boolean value
    beard_bool = msft_pred or amzn_pred or amzn_high_confd or msft_high_cofd 
    bald_bool = low_hair_ratio or high_bald_microsoft or combined_result

    imgpath_toadd = path.join(img_path,group,items[0])
    
    return imgpath_toadd,beard_bool,bald_bool

#this function is to change extension from ".JPG" to ".jpg"
def change_extension(path):
    path = Path(path)
    path = path.with_suffix('')
    path = path.with_suffix('.jpg')

    return path


def filtering(filepath,img_path,group,type):
    attributes = np.loadtxt(filepath,delimiter = " | " ,dtype="str")
    facial_hair =[]
    no_facial_hair =[]
    bald =[]
    not_bald=[]
    beard_bald=[]
    nobeard_bald=[]
    beard_notbald=[]
    nobeard_notbald=[]

    if type == "beard":
        for items in tqdm(attributes):
            imgpath_toadd,beard_bool,bald_bool = get_attribute_values(items,img_path,group)

            #comment this if the extension of the file is ".JPG"
            #imgpath_toadd = change_extension(imgpath_toadd)

            if path.exists(imgpath_toadd):
                if beard_bool:
                    facial_hair.append(imgpath_toadd)
                else:
                    no_facial_hair.append(imgpath_toadd)
            

        print(len(facial_hair),len(no_facial_hair))

        return facial_hair,no_facial_hair

    elif type == "bald":
        for items in tqdm(attributes):
            imgpath_toadd,beard_bool,bald_bool = get_attribute_values(items,img_path,group)

            #comment this if the extension of the file is ".JPG"
            #imgpath_toadd = change_extension(imgpath_toadd)

            if path.exists(imgpath_toadd):
                if bald_bool:
                    bald.append(imgpath_toadd)
                else:
                    not_bald.append(imgpath_toadd)
        
        print(len(bald),len(not_bald))
        return bald,not_bald
    
    else:
        not_found = 0
        for items in tqdm(attributes):
            imgpath_toadd,beard_bool,bald_bool = get_attribute_values(items,img_path,group)
            
            #comment this if the extension of the file is ".JPG"
            #imgpath_toadd = change_extension(imgpath_toadd)

            if path.exists(imgpath_toadd):
                if beard_bool and bald_bool:
                    beard_bald.append(imgpath_toadd)
                elif beard_bool and (not bald_bool):
                    beard_notbald.append(imgpath_toadd)
                elif (not beard_bool) and (not bald_bool):
                    nobeard_notbald.append(imgpath_toadd)
                else:
                    nobeard_bald.append(imgpath_toadd)
            else:
                not_found +=1
        
        print("The number of items that were not found were : {}".format(not_found))

        print(len(beard_bald),len(beard_notbald),len(nobeard_bald),len(nobeard_notbald))

        return beard_bald,beard_notbald,nobeard_bald,nobeard_notbald
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get imagelist after beard or bald or beard + bald filtering")
    parser.add_argument("--source", "-s", help="hair attribute path")
    parser.add_argument("--dest", "-d", help="Folder to save the imagelist")
    parser.add_argument("--group", "-gr", help="C_M for Caucasian male and such")
    parser.add_argument("--imgpath","-img", help = "Main directory where the images are")
    parser.add_argument("--category","-c", help = "Pass one of these arguments as needed. Takes - beard, bald and both")
    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    assert args.category in ["bald","beard","both"]

    if args.category == 'beard':
        facial_hair,no_facial_hair = filtering(args.source,args.imgpath,args.group,args.category)
        np.savetxt(path.join(args.dest,"{}_facial_hair.txt".format(args.group)), facial_hair, delimiter = " ",fmt='%s',newline='\n')
        np.savetxt(path.join(args.dest,"{}_no_facial_hair.txt".format(args.group)), no_facial_hair, delimiter = " ",fmt='%s',newline='\n')

    elif args.category == 'bald':
        facial_hair,no_facial_hair = filtering(args.source,args.imgpath,args.group,args.category)
        np.savetxt(path.join(args.dest,"{}_bald.txt".format(args.group)), facial_hair, delimiter = " ",fmt='%s',newline='\n')
        np.savetxt(path.join(args.dest,"{}_not_bald.txt".format(args.group)), no_facial_hair, delimiter = " ",fmt='%s',newline='\n')

    elif args.category == 'both':
        beard_bald,beard_notbald,nobeard_bald,nobeard_notbald = filtering(args.source,args.imgpath,args.group,args.category)
        np.savetxt(path.join(args.dest,"{}_beard_bald.txt".format(args.group)), beard_bald, delimiter = " ",fmt='%s',newline='\n')
        np.savetxt(path.join(args.dest,"{}_beard_notbald.txt".format(args.group)), beard_notbald, delimiter = " ",fmt='%s',newline='\n')
        np.savetxt(path.join(args.dest,"{}_nobeard_bald.txt".format(args.group)), nobeard_bald, delimiter = " ",fmt='%s',newline='\n')
        np.savetxt(path.join(args.dest,"{}_nobeard_notbald.txt".format(args.group)), nobeard_notbald, delimiter = " ",fmt='%s',newline='\n')
    
