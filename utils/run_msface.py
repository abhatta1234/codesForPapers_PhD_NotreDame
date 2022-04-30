#Orginal Author: Vítor Albiero
import json, os, requests
import cv2
import argparse
from os import path, makedirs
import pandas as pd
import numpy as np
from tqdm import tqdm

subscription_key = #"You can get a free subscription key that works for a month"

face_api_url = 'https://ndface.cognitiveservices.azure.com/' + '/face/v1.0/detect'

headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'

params = {
    # Request parameters
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': attributes,
    'recognitionModel': 'recognition_03',
    'returnRecognitionModel': 'false',
    'detectionModel': 'detection_01',
}

def run_api(source, output_folder):
    img_list = np.asarray(pd.read_csv(source, delimiter=' ', header=None)).squeeze()
    fail = []

    for img_path in tqdm(img_list):
        image_label = path.join(*img_path.split('/')[-2:])
        output_path = path.join(output_folder, image_label[:-3] + 'json')

        subject_path = path.split(output_path)[0]
        if not path.exists(subject_path):
            makedirs(subject_path)

        img = cv2.imread(img_path)
        img_encode = cv2.imencode('.png', img)[1]   # encode the image

        response = requests.post(face_api_url, params=params, 
                headers=headers, data=(img_encode.tobytes()))

        if not response.ok:
            print(response)
            fail.append(img_path)
            continue

        file = open(output_path, 'w')
        json.dump(response.json(), file)
        file.close()

    np.savetxt(path.split(source)[1][:-4]+'_fail.txt', fail, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Microsoft Face API.')
    parser.add_argument('--source', '-s', help='File with a list of images.')
    parser.add_argument('--output_folder', '-o', help='Output folder.')

    args = parser.parse_args()

    if not path.exists(args.output_folder):
        makedirs(args.output_folder)

    run_api(args.source, args.output_folder)
