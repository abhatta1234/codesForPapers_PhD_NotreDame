All the results shown in this paper can be reproduced using the instruction as shown:

# Preliminary Steps

### Get the Hair Ratio using BiSeNet

We first get the BiSeNet hair ratio for the group considered. To obtain the BiSeNet mask for images, you can use [Vitor Albiero's repo](https://github.com/vitoralbiero/face-parsing.PyTorch) to process all the images at once. Now, once you have the BiSeNet mask, the following code can be used to get hair ratio

~~~bash
python3 utils/bisenet_parsing.py -s BiSeNet_results_folder -d dest_to_hairRatio_txt -n file_save_name
~~~

Example usage:

~~~bash
python3 bisenet_parsing.py -s ../BiSeNet/C_M/ -d ../BiSeNet/hair_ratio/ -n C_M_hair_ratio
~~~

### Get Amazon Rekognition and MS Face API

More details about [Amazon Rekognition](https://docs.aws.amazon.com/managedservices/latest/userguide/rekognition.html) and [MS Face](https://azure.microsoft.com/en-us/services/cognitive-services/face/). If needed, helper functions to use these API can be found in [utils](https://github.com/abhatta1234/Hair_Dimensions_Balancing/tree/main/utils) folder.

### Compile BiSeNet and Face API to one txt

Here we compile all the results(per group) to a separate text that can be used later for thresholding and attribute balancing across groups considered. The following code can be used to get the results

~~~bash
python3 hair_attribute_compilation.py -amzn path_to_amzn_results -msft path_to_amzn_results \
-bise path_to_BiSeNet_hair_ratio_txt -dest destination -n save_name
~~~

Example usage:
~~~bash
python3 hair_attribute_compilation.py -amzn ../Amazon/C_M/ -msft ../MSFace/API/C_M/ \
-bise ../BiSeNet/hair_ratio/C_M_hair_ratio.txt -dest ../attribute_list/ -n C_M_attribute_list
~~~

# Thresholding

Now that we have all the attribute list compiled, we can use it for thresholding. Of course, you could set up it however, I decided to do this way considering other aspects of the research/experiments.<br>

The following code can be used to do the thresholding. The thresholding scheme is described in the [paper]() and the example usage shows the filtering for both no facial hair and no baldness.

~~~bash
python3 thresholding.py -s path_to_compiled_attribute_list -d destination_to_save_txt \
-gr group_name -img main_path_where_imgs_are_stored -c category[options:beard,bald,both]
~~~
Example usage:
~~~bash
python3 codes/thresholding.py -s ../attribute_list/C_F_hair_attributes.txt -d ../thresholding_results/ \
-gr C_F -img /afs/crc.nd.edu/user/a/abhatta/MORPH3/ -c both
~~~

# Hair-Style Balancing

Now that we have filtered the images to not have facial hair or baldness, we run IoU(Intersection Over Union) to find image pair with best hair match across the group.The IoU computation can be done as shown:
~~~bash
python3 codes/equalize_hair_iou.py -ga BiSeNet_maskpath_txt_group1 -gb BiSeNet_maskpath_txt_group2 \
-o save_destination -d name_of_dataset -t IoU_thresholding_value
~~~
Example Usage:
~~~bash
python3 equalize_hair_iou.py -ga maskpath/C_M_nobeard_maskpath.txt -gb maskpath/C_F_maskpath.txt \
-o iou/C_M_vs_C_F/ -d MORPH -t 0.7
~~~
Note that, IoU computation code takes txt list of BiSeNet Segmenation maskpath and return the maskpath aswell. To create maskpath list from imglist and vice-versa can be found in [utils](https://github.com/abhatta1234/Hair_Dimensions_Balancing/tree/main/utils) folder. 

At this point you should have txt files with imagepath list for two-groups that are roughly hair-style balanced. In the context of this [paper](), it was done to balance different dimensions of hair(baldness, facial hair and hair-style) across male and female. 

You can extract the ArcFace feature vector for these images, match those feature vector and plot impostor and genuine distribution as shown in the paper following [Vitor Albiero's repo](https://github.com/vitoralbiero/face-parsing.PyTorch). You can always reach out to me at this [abhattand@gmail.com](www.gmail.com) if you need the exact pre-trained model used for this paper. The remaining steps are clearly explained by Vítor Albiero in his repo. 


# BootStrap Confidence Analysis Part

After you balance across two groups for hair-dimension, the size of the subject count/images that you compare with reduces by a significant amount. So, doing boostrap confidence analysis ensures that the results are not just by a random chance. 

~~~bash
python3 codes/random_pick.py -s all_images_features_txtfile -d save_dest -sub num_subject -img num_images -itr num_iterations
~~~

Example usage:
~~~bash
python3 ../codes/random_pick.py -s features/arcface/all/C_M.txt -d featurestxt/arcface/C_M/ -sub 178 -img 344 -itr 1000
~~~
Based on this example usage, 1000 txtfiles of C_M with 178 subjects and 344 distinct images in each of the txtfiles will be saved. 

# D-prime Calculations

The dprime calculations can be done shown below:

Example usage:
~~~bash
python3 codes/delta_dprim_compute.py \
-a1 ../match/gender_balanced/all/C_M/CM_authentic.npy -i1 ../match/gender_balanced/all/C_M/CM_impostor.npy -l1 C_M \
-a2 ../match/gender_balanced/all/C_F/CF_authentic.npy -i2 ../match/gender_balanced/all/C_F/CF_impostor.npy -l2 C_F \
~~~

Note that Vítor Albiero's repo the plotting is done differently. A modified .py file to plot the impostor and genuine curves exactly as shown in the paper can be found in codes folder


