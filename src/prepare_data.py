import argparse
import os 
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import argparse


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False


def get_annotations(path):
    annotations = []
    images = []
    for txt_file in glob(path + '/*.txt'):
        annotations.append(txt_file)
        image = txt_file.replace('txt', 'jpg')
        images.append(image)

    return annotations, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to images')
    opt = parser.parse_args()

    annotations, images = get_annotations(opt.path)

    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    os.system('mkdir data/images data/annotations')
    os.system('mkdir data/images/train data/images/val data/images/test data/annotations/train data/annotations/val data/annotations/test')

    move_files_to_folder(train_images, 'data/images/train')
    move_files_to_folder(val_images, 'data/images/val/')
    move_files_to_folder(test_images, 'data/images/test/')
    move_files_to_folder(train_annotations, 'data/annotations/train/')
    move_files_to_folder(val_annotations, 'data/annotations/val/')
    move_files_to_folder(test_annotations, 'data/annotations/test/')

    os.system('cp -r data/annotations data/labels')


if __name__ == "__main__":
    main()
