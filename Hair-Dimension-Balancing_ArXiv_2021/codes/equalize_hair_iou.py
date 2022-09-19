# Original Author: Vitor Albiero
# Modified as Needed: Aman Bhatta

import argparse
import os
from datetime import datetime
from multiprocessing import Pool
from os import makedirs, path

import cv2
import numpy as np
from tqdm import tqdm


class hairOverlap():
    def __init__(self, group_a_path, group_b_path, dataset_name, mask_path, threshold):
        print(f'hair IoU between {group_a_path} to {group_b_path}')
        # length of ids to get from image files
        self.id_length = -1
        self.dataset_name = dataset_name
        self.threshold = threshold

        self.mask = None
        if mask_path is not None:
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        group_a_file = np.sort(np.loadtxt(group_a_path, dtype=str))
        group_b_file = np.sort(np.loadtxt(group_b_path, dtype=str))
        assert len(group_a_file) >= len(group_b_file)

        # load images, subject ids, image labels from group_a file
        self.group_a_file_name = path.split(group_a_path)[1]

        self.group_a, self.group_a_ids, self.group_a_labels = self.get_images(group_a_file)

        self.group_b_file_name = path.split(group_b_path)[1]
        self.group_b, self.group_b_ids, self.group_b_labels = self.get_images(group_b_file)

        self.group_a_selected = []
        self.group_b_selected = []
        self.pre_selected = []
        self.selected = {}

    def get_images_label(self, image_path):
        subject_id = path.split(image_path)[1]
        image_label = path.join(path.split(path.split(image_path)[0])[1], subject_id[:-4])
        

        if self.dataset_name == 'CHIYA':
            subject_id = subject_id[:-5]

        elif self.dataset_name == 'CHIYA_VAL':
            subject_id = image_label[1:-4]

        elif self.dataset_name == 'PUBLIC_IVS':
            subject_id = path.split(image_label)[0]

        elif self.id_length > 0:
            subject_id = subject_id[:self.id_length]
        else:
            subject_id = subject_id.split('_')[0]

        return subject_id, image_path

    def get_images(self, file):
        all_images = []
        all_labels = []
        all_subject_ids = []

        for j in tqdm(range(len(file))):
            image_path = file[j]
            image = self.get_image(image_path)
            subject_id, image_label = self.get_images_label(image_path)

            all_images.append(image)
            all_subject_ids.append(subject_id)
            all_labels.append(image_label)

        return np.asarray(all_images), np.asarray(all_subject_ids), np.asarray(all_labels)

    def get_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Hair
        # img[np.logical_and(img > 0, img <= 16)] = 0
        # img[img > 16] = 1

        img[np.logical_not(img==17)] = 0
        img[img == 17] = 1

        if self.mask is not None:
            img = cv2.resize(img, (224, 224))
            img[self.mask == 0] = 0

        img = cv2.resize(img, (112, 112))

        return img.flatten()

    def run(self):
        for idx in tqdm(range(len(self.group_a))):
            total_hair = self.group_a[idx] + self.group_b
            total_hair[total_hair == 2] = 1
            total_hair = np.sum(total_hair, axis=1)
            matching_hair = np.sum(self.group_a[idx] * self.group_b, axis=1)

            iou = matching_hair / total_hair
            j = np.argmax(iou)

            if iou[j] < self.threshold:
                continue

            self.pre_selected.append([self.group_a_labels[idx], self.group_b_labels[j], iou[j]])

    def select(self):
        self.pre_selected = np.asarray(self.pre_selected)
        self.pre_selected = self.pre_selected[self.pre_selected[:, 2].astype(float).argsort()][::-1]
        for idx in tqdm(range(len(self.pre_selected))):

            if self.pre_selected[idx, 1] not in self.selected:
                if self.pre_selected[idx, 0] not in self.selected:
                    self.selected[self.pre_selected[idx, 0]] = True
                    self.selected[self.pre_selected[idx, 1]] = True

                    self.group_a_selected.append(self.pre_selected[idx, 0])
                    self.group_b_selected.append(self.pre_selected[idx, 1])

    def run_parallel(self):
        pool = Pool(os.cpu_count() - 2)
        print(os.cpu_count())

        indices = np.linspace(0, len(self.group_a) - 1, len(self.group_a)).astype(int)
        for result in pool.map(self.get_iou, indices):
            if result is not None:
                self.pre_selected.append(result)

    def get_iou(self, idx):
        #print("groupa,groupb",self.group_a[idx].shape,self.group_b.shape)
        total_hair = self.group_a[idx] + self.group_b
        
        total_hair[total_hair == 2] = 1
        total_hair = np.sum(total_hair, axis=1)
        
        matching_hair = np.sum(self.group_a[idx] * self.group_b, axis=1)
        #print("total hair ||||| matching hair",total_hair.shape,matching_hair.shape,np.count_nonzero(total_hair),np.count_nonzero(matching_hair))
        
        iou = matching_hair / total_hair
        #print(iou)
        j = np.argmax(iou)

        if iou[j] < self.threshold:
            return None

        return [self.group_a_labels[idx], self.group_b_labels[j], iou[j]]

    def save_selected_images(self, output):
        print(len(self.group_a_selected),len(self.group_b_selected))
        np.savetxt(path.join(output, self.group_a_file_name), self.group_a_selected, delimiter=' ', fmt='%s')
        np.savetxt(path.join(output, self.group_b_file_name), self.group_b_selected, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matches Two Lists Based on hair IoU')
    parser.add_argument('-group_a', '-ga', help='Group A image list.')
    parser.add_argument('-group_b', '-gb', help='Group B image list.')
    parser.add_argument('-output', '-o', help='Output folder.')
    parser.add_argument('-dataset', '-d', help='Dataset name.')
    parser.add_argument('--mask', '-m', help='Mask.')
    parser.add_argument('--threshold', '-t', help='Threshold', default=0.9)

    args = parser.parse_args()

    if not path.exists(args.output):
        makedirs(args.output)

    hair_overlap = hairOverlap(args.group_a, args.group_b, args.dataset.upper(),
                               args.mask, float(args.threshold))
    time1 = datetime.now()
    hair_overlap.run_parallel()
    time2 = datetime.now()
    print(time2 - time1)
    hair_overlap.select()
    hair_overlap.save_selected_images(args.output)
