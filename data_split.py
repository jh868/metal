import os
import glob
import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def data_split_by_json(dir_path):
    image_path_list = glob.glob(os.path.join(dir_path, 'images', '*'))
    # print(len(image_name_list))

    json_file_path = os.path.join(dir_path, 'anno', 'annotation.json')
    json_data = None
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
        # print(json_data)

    if json_data is None:
        print('read json fail')
        return

    temp_dict = crop_and_classify_by_label(image_path_list, json_data)
    # print(temp_dict)

    for _, data in temp_dict.items():
        train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True, random_state=20230116)
        # print(len(train_data), len(test_data))

        temp_dict = {
                'train': train_data,
                'test': test_data
            }

        for task, data in temp_dict.items():
            copy_data(task, data)
        

def copy_data(task, data):
    for index, image_dict in enumerate(data):
        # print(image_dict)
        # exit()

        label = image_dict['label']
        temp_image = image_dict['image']
        image = make_square_image(temp_image)

        dir_path = os.path.join('./', 'dataset', task, label)
        dest_path = os.path.join(dir_path, '{}_{}.png'.format(label, index))

        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(dest_path, image)


def crop_and_classify_by_label(image_path_list, json_data):
    temp_dict = dict()

    for image_path in image_path_list:
        image_name = os.path.basename(image_path)

        image_object = json_data[image_name]

        anno_list = image_object['anno']

        for item in anno_list:
            label = item['label']
            bbox = item['bbox']
            # print(label, bbox)

            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            # print(xmin, xmax, ymin, ymax)

            image = cv2.imread(image_path)
            crop_image = image[ymin:ymax, xmin:xmax, :]
            # print(crop_image.shape)
            # cv2.imshow('test', crop_image)
            # cv2.waitKey(0)
            # exit()

            if label not in temp_dict.keys():
                temp_dict[label] = list()
            else:
                temp_dict[label].append({
                    'label': label,
                    'image': crop_image
                })
    # print(temp_dict)

    return temp_dict


def make_square_image(origin_image):
    try:
        height, width, channels = origin_image.shape
    except:
        print('not image', '#' * 10)
        return
    
    long_side = height if height > width else width
    long_side = 224 if 224 > long_side else long_side

    x, y = long_side, long_side
    
    squre_image = np.zeros((x, y, channels), np.uint8)
    squre_image[int((y - height) / 2):int(y - (y - height) / 2), 
                int((x - width) / 2):int(x - (x - width) / 2)] = origin_image
    
    return squre_image
        

"""
- 2. metal_damaged_detection
    - anno
        - annotation.json
    - images
        - ...jpg
- code
    - data_split.py <-- (current location)
"""

file_path = os.path.join('./', '2. metal_damaged_detection')
data_split_by_json(file_path)