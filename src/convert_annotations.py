import torch
from IPython.display import Image  # for displaying images
import os
import random
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"hand": 0}
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))


# Function to get the data from annotations
def convert_info(anno_file):
    with open(anno_file) as f:
        lines = f.readlines()

        dicts = []
        filenames = []
        # Initialise the info dict 
        info_dict = {}
        info_dict['bboxes'] = []
        filename = ''

        for line in lines:
            line_split = line.split(',')
            temp_filename = line_split[0]
            if temp_filename != filename:
                if len(info_dict['bboxes']) != 0:
                    dicts.append(info_dict)
                info_dict = {}
                info_dict['bboxes'] = []
                filename = temp_filename

            filename = line_split[0]
            filenames.append(filename)
            info_dict['filename'] = filename
            info_dict['image_size'] = tuple([480, 360, 3])
            bbox = {}
            bbox["class"] = 'hand'
            # xmin, xmax, ymin, ymax
            bbox['xmin'] = int(line_split[1])
            bbox['xmax'] = int(line_split[2])
            bbox['ymin'] = int(line_split[3])
            bbox['ymax'] = int(line_split[4])
            info_dict['bboxes'].append(bbox)

    return dicts, filenames


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolo(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("data/coco/COCO-Hand-S_Images", info_dict["filename"].replace("jpg", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to images')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotations txt')
    parser.add_argument('--plot', action='store_true', help='To plot converted bboxes on a random image')
    opt = parser.parse_args()

    dicts, filenames = convert_info(opt.annotations)
    for info_dict in tqdm(dicts):
        convert_to_yolo(info_dict)

    if opt.plot:
        # Get any random annotation file
        annotation_file = random.choice(filenames)
        with open(opt.images + '/' + annotation_file.replace("jpg", "txt"), "r") as file:
            annotation_list = file.read().split("\n")[:-1]
            annotation_list = [x.split(" ") for x in annotation_list]
            annotation_list = [[float(y) for y in x ] for x in annotation_list]

        #Get the corresponding image file
        image_file = opt.images + '/' + annotation_file
        assert os.path.exists(image_file)

        #Load the image
        image = Image.open(image_file)

        #Plot the Bounding Box
        plot_bounding_box(image, annotation_list)


if __name__ == "__main__":
    main()
