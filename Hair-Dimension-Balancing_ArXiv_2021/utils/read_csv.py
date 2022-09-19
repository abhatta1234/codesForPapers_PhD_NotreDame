import shutil
from os import path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

info = "../csv_folder/bxgrid-results.csv"


# comma delimited is the default
df = pd.read_csv(info, low_memory=False)

# remove the non-numeric columns
# df = df._get_numeric_data()
subjects = df[["subjectid"]]
gender = df[["gender"]]
race = df[["race"]]
yob = df[["YOB"]]
date = df[["date"]]
shotid = df[["shotid"]]
sequenceid = df[["sequenceid"]]
iformat = df[["format"]]
emotion = df[["emotion"]]
weather = df[["weather"]]
tags = df[["tag_list"]]
sensorid = df[["sensorid"]]

subjects = subjects.values.tolist()
gender = gender.values.tolist()
race = race.values.tolist()
yob = yob.values.tolist()
date = date.values.tolist()
sequenceid = sequenceid.values.tolist()
shotid = shotid.values.tolist()
iformat = iformat.values.tolist()
weather = weather.values.tolist()
tags = tags.values.tolist()
sensorid = sensorid.values.tolist()

sensor_id_collection = []
save_txt = []

sensors = ['lg4000diff25' 'lg4000diff30' 'nd1N00001' 'nd1N00002' 'nd1N00005'
 'nd1N00010' 'nd1N00012' 'nd1N00013' 'nd1N00014' 'nd1N00020' 'nd1N00040'
 'nd1N00046' 'nd1N00052' 'nd1N00053' 'nd1N00055' 'nd1N00056' 'nd1N00058'
 'nd1N00062' 'nd1N00063' 'nd1N00065' 'nd1N00066' 'nd1N00067' 'nd1N00075'
 'nd1N00076' 'nd1N00078']

sensors_dict = {}



for i in tqdm(range(len(shotid))):
    # i_format = iformat[i][0]

    # if i_format == "jpeg":
    i_format = "npy"
    

    if "moustache" in str(tags[i][0]).lower():
        tag = "Moustache"
    else:
        tag = "None"

    if "hat" in str(tags[i][0]).lower() or "closed" in str(tags[i][0]).lower():
        continue

    if str(sensorid[i][0]) in ["nd1N00072", "nd1N00069", "lg4000diff20"]:
        continue

    sensor_id_collection.append(sensorid[i][0])


    new = "{}_{}_{}_{}_{}_{}_{}_{}.{}".format(
            subjects[i][0],
            gender[i][0],
            race[i][0],
            yob[i][0],
            date[i][0][0:10],
            # shotid[i][0],
            sequenceid[i][0],
            weather[i][0],
            tag,
            i_format.lower())
    

    # if gender[i][0] == "Male" and weather[i][0] == "Inside":
    #     print("/afs/crc.nd.edu/user/a/abhatta/IJCB_paper/complete_pipeline/features/arcface/all/C_M/" + new)
    #     tosavepath = "/afs/crc.nd.edu/user/a/abhatta/IJCB_paper/complete_pipeline/features/arcface/all/C_M/" + new
    #     if path.exists(tosavepath):
    #         if sensorid[i][0] in sensors_dict:
    #             sensors_dict[sensorid[i][0]] +=1
    #         else:
    #             sensors_dict[sensorid[i][0]] = 1
    #         save_txt.append(tosavepath)
    #     else:
    #         print("Not found: ",tosavepath)


    if sensorid[i][0] == 'nd1N00063' and gender[i][0] == "Male" and weather[i][0] == "Inside":
        #print("/afs/crc.nd.edu/user/a/abhatta/IJCB_paper/complete_pipeline/features/arcface/all/C_M/" + new)
        tosavepath = "/afs/crc.nd.edu/user/a/abhatta/IJCB_paper/complete_pipeline/features/arcface/all/C_M/" + new
        if path.exists(tosavepath):
            save_txt.append(tosavepath)
        else:
            print("Not found: ",tosavepath)
  
print(len(save_txt))
np.savetxt("{}.txt".format("../separate_sensors/CM_nd1N00063"),save_txt,fmt="%s",newline='\n')  



# sensor_id_collection = np.array(sensor_id_collection)
# unique,counts =  np.unique(sensor_id_collection, return_counts=True)
# for i in range(len(unique)):
#     print(unique[i],counts[i])

# if path.isfile(old):
#     shutil.move(old, new)
# img = cv2.imread(old)
# cv2.imwrite(new, img)
# os.rename(old, new)
# else:
#     print("File {} not found!".format(old))


'''
'nd1N00012': 632, 'nd1N00013': 140, 'nd1N00014': 7, 'nd1N00002': 3042, 'nd1N00040': 892, 'nd1N00005': 299, 
'nd1N00001': 334, 'nd1N00046': 44, 'nd1N00053': 5, 'nd1N00052': 18, 'nd1N00056': 4, 'nd1N00067': 12, 'nd1N00063': 759, 
'nd1N00066': 1, 'nd1N00058': 5, 'nd1N00065': 8, 'nd1N00062': 1, 'lg4000diff25': 2, 'nd1N00076': 794, 'nd1N00075': 1, 'nd1N00055': 3
'''

# Three highest peaks
'''
'nd1N00002': 3042
'nd1N00040': 892
'nd1N00063': 759
'''