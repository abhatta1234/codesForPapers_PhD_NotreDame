#Original Author : Vítor Albiero
import boto3
import json
import numpy as np
import argparse
from os import path, makedirs
from tqdm import tqdm


def run_api(source, output_folder):
    client = boto3.client('rekognition')

    img_list = np.loadtxt(source, dtype=str)

    for img_path in tqdm(img_list):
        image = open(img_path, 'rb')

        image_name = path.split(img_path)[1]
        subject = path.split(path.split(img_path)[0])[1]
        output_path = path.join(output_folder, subject, image_name[:-3] + 'json')

        if not path.exists(path.split(output_path)[0]):
            makedirs(path.split(output_path)[0])

        if path.isfile(output_path):
            print('Image {} already processed, skipping...'.format(image_name))
            continue

        response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])

        file = open(output_path, 'w')
        json.dump(response, file)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Amazon Rekognition.')
    parser.add_argument('--source', '-s', help='File with a list of images.')
    parser.add_argument('--output_folder', '-o', help='Output folder.')

    args = parser.parse_args()

    if not path.exists(args.output_folder):
        makedirs(args.output_folder)

    run_api(args.source, args.output_folder)
