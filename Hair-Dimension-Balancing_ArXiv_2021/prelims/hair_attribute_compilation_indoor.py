from unittest import result
import numpy as np
import os
from torch import empty
from tqdm import tqdm
from pathlib import Path
import json
import argparse



def parse_hair_microsoft(msft_directory,bisenet_txt):
    attributes = {}

    # Hair ratio from BiSeNet
    hair_ratio = dict(np.loadtxt(bisenet_txt,dtype="str"))

    #Get the number of sub_directories - needed for naming later
    num_subdirectory = len(next(os.walk(msft_directory))[1])

    # microsoft parsing
    for subdir, dirs, files in os.walk(msft_directory,topdown=True):
        for filename in tqdm(files):
            #incase if the folder is divided into various subfolders
            #print(subdir)
            filepath = os.path.join(subdir,filename)
            #print(filepath)
            with open(filepath, 'r') as j:
                contents = json.loads(j.read())
                if len(contents)>0:
                    path = Path(filepath)
                    if num_subdirectory:
                        img_name = os.path.join(path.parent.stem,path.stem)+ ".JPG"
                        hair_ratio_name = path.stem + ".JPG"
                    else:
                        img_name = path.stem + ".JPG"
                        hair_ratio_name = path.stem + ".JPG"

                    hair_details = contents[0]["faceAttributes"]["hair"]
                    attributes[img_name]=[]
                    attributes[img_name].append({k: v for k, v in hair_details.items() if k == "bald" or k == "invisible"})
                    
                    #Hair Ratio BiSeNet
                    attributes[img_name].append(hair_ratio[hair_ratio_name])
                    
                    # facial_hair attribute
                    attributes[img_name].append(contents[0]["faceAttributes"]["facialHair"])
                              
    return attributes


def parse_amzn_final(amzn_directory,msft_directory, bisenet_txt):
    # parsing microsoft results
    print("parsing microsoft!!")
    attributes = parse_hair_microsoft(msft_directory,bisenet_txt)

    #Get the number of sub_directories - needed for naming later
    num_subdirectory = len(next(os.walk(msft_directory))[1])
    print("parsing amazon!!")
    # Amazon Parsing
    no_data = []
    for subdir, dirs, files in os.walk(amzn_directory,topdown=True):
        for file in tqdm(files):
            json_name = os.path.join(subdir+file)
            file = open(json_name, 'r')
            response = json.load(file)
            if len(response['FaceDetails']) > 0:
                path = Path(json_name)

                if num_subdirectory:
                    img_name = os.path.join(path.parent.stem,path.stem)+ ".JPG"
                else:
                    img_name = path.stem + ".JPG"
                    
                bool_beard = response['FaceDetails'][0]['Beard']['Value']
                confd_val = response['FaceDetails'][0]['Beard']['Confidence']

                if str(img_name) in attributes:
                    # Amzn Beard
                    attributes[str(img_name)].append([bool_beard,confd_val])
                else:
                    no_data.append(img_name)
    return [attributes,no_data]

def save_txt(dict_results,save_name):
    save_array = []
    for i in dict_results:
        if i.split("_")[-2] == "Inside":
            save_array.append([i ,dict_results[i][0], dict_results[i][1],dict_results[i][2],dict_results[i][3]])
    head = 'img_name,bald_msft,bisenet_ratio,beard_msft,beard_amzn'
    np.savetxt(save_name, save_array, delimiter = " | " , header=head,fmt='%s',newline='\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the attribute txt for thresholding")
    parser.add_argument("--amazon", "-amzn", required = True, help="Path to Amazon directory")
    parser.add_argument("--msft", "-msft", required = True, help="Path to Microsoft directory")
    parser.add_argument("--bisenet", "-bise", required = True, help="Path to BiSeNet hair ratio txt")
    parser.add_argument("--destination", "-dest",help="Path to destination folder")
    parser.add_argument("--name", "-n", required = True, help="Name to save the file.")
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    
    save_name = os.path.join(args.destination,args.name)

    print("Parsing Started!!!")

    parsing_result = parse_amzn_final(amzn_directory = args.amazon ,msft_directory=args.msft,bisenet_txt=args.bisenet)

    result_dict = parsing_result[0]
    empty_list = parsing_result[1]

    save_name = os.path.join(args.destination,args.name) +".txt"
    print("saving files at {}".format(save_name))
    save_txt(result_dict,save_name)
    print("The number of faces that were not detected were: ", len(empty_list))

