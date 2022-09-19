
# This codes takes a txt files and saves the images/features that are inside 

import numpy as np

path = '../maskpath/C_M_nobeard_nobald_nobeard_notbald_maskpath.txt'

check = np.loadtxt(path,dtype=str)
print(len(check))
count = 0 
subset = []
for items in check:
    intmdt = items.split("_")[-3]
    if intmdt == "Inside":
        print(intmdt,items)
        count+=1
        subset.append(items)
print(len(subset))

save_name = "../maskpath/indoor_C_M_all_balanced"
np.savetxt("{}.txt".format(save_name),subset,fmt='%s',newline='\n')