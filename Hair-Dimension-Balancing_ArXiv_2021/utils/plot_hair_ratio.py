import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def plot_gender(dist1,label1,dist2,label2):
    plt.rcParams["figure.figsize"] = [8, 6]
    matplotlib.rc('axes', linewidth=2)
    
    plt.rcParams["font.size"] = 20
    
    #plt.gca().set_yscale("log")
    
    plt.hist(dist1.astype(float),label=label1,linewidth=1.5,bins="auto",density=True,color="b",histtype="step",)

    plt.hist(dist2.astype(float),label=label2,linewidth=1.5,bins="auto",density=True, color="r",histtype="step",)
    
    plt.legend(loc="upper right", fontsize = 16)
                
    plt.ylabel("Relative Frequency")
    plt.xlabel("Hair Ratio")

    plt.tight_layout(pad=0.2)
    
    #plt.title("Caucasian Hair Distribution",fontsize=20,fontweight='bold',color='black')

    
    plt.savefig("C_M_vs_C_F_hair_ratio_inside", dpi=300,bbox_inches='tight')


# dist1 = np.asarray(list(CM_Bisenet.values()))
# dist2 = np.asarray(list(CF_Bisenet.values()))
# plot_gender(dist1=dist1,label1="AA_M",dist2=dist2,label2="AA_F")

male = np.loadtxt("../BiSeNet/hair_ratio/C_M_hair_ratio.txt",dtype=str)
female = np.loadtxt("../BiSeNet/hair_ratio/C_F_hair_ratio.txt",dtype=str)


# This section to get the hair ratio plot for inside images
male_inside = []
female_inside = []

for items in male:
    if items[0].split("_")[-2] == "Inside":
        male_inside.append(items[1])

for items in female:
    if items[0].split("_")[-2] == "Inside":
        female_inside.append(items[1])


male_ratio = np.asarray(male_inside).astype(float)
female_ratio = np.asarray(female_inside).astype(float)


# male_ratio = np.asarray(male[:,1]).astype(float)
# female_ratio = np.asarray(female[:,1]).astype(float)

plot_gender(dist1=male_ratio,label1="C_M",dist2=female_ratio,label2="C_F")
