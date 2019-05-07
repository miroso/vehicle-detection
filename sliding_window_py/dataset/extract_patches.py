#!/home/miro/virtualenv/env_cv4/bin/python
# coding: utf-8
"""
Extracts square image patches from the dataset.

run file:
python extract_patches.py
-i /media/hd1/datasets/kitty/data_object_image_2/training/image_2
-l /media/hd1/datasets/kitty/data_object_label_2/training/label_2
-o /home/miro/workspace/projects/vehicle-detection/sliding_window_py/dataset
-ps 64
"""

import argparse
import math
import sys
from pathlib import Path
import cv2

def main():
    """ Extracts square image patches from the dataset

    Raises:
        Exception: Python 3 version is required
    """
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='Extracts square image patches and sorts them into respective folders')
    parser.add_argument('-i', '--image_file_path',
                        required=True,
                        help='Absolute path to directory with input images')
    parser.add_argument('-l', '--label_file_path',
                        required=True,
                        help='Absolute path to a directory with label files for each image')
    parser.add_argument('-o', '--output_path',
                        required=True,
                        help='Absolute path to the output folder, to store created quadratic patches and label files')
    parser.add_argument('-ps', '--patch_size',
                        required=False,
                        default=32,
                        help='Width and height of the resulting patches, defaults to 32')

    args = parser.parse_args()
    process_images(args.image_file_path, args.label_file_path, args.output_path, int(args.patch_size))


def process_images(image_file_path, label_file_path, output_path, patch_size):
    """Extract patches from all dataset images.

    Arguments:
        image_file_path {str} -- absolute path to the directory with input images.
        label_file_path {str} -- absolute path to a diretory with label files
        output_path {str} -- absolute path to a directory where image patches will be stored.
        patch_size {int} -- size of patches
    """
    image_files = Path(image_file_path).glob('**/*.png')
    label_files = Path(label_file_path).glob('**/*.txt')
    assert image_files, 'Missing Images'
    assert label_files, 'Missing labels'

    output_image_path = Path(output_path, 'patches')
    output_label_path = Path(output_path, 'labels')
    create_output_folders(output_image_path, output_label_path)

    for image_path in image_files:
        label_path = Path(label_file_path, image_path.name.split('.')[0] + '.txt')
        extract_patch_data(image_path, label_path, patch_size, output_path)


def extract_patch_data(image_path, label_path, patch_size, output_path):
    """Extract patches and label information from the provided image.

    Arguments:
        image_path {str} -- path to image to extract patches from
        label_path {str} -- patch to label file containing label information for all boxes in image
        patch_size {str} -- desired size of patches
        output_path {str} -- patches and labels will be stored here in separate directories
    """
    file_name = image_path.name
    output_image_path = Path.joinpath(Path(output_path, "patches"), image_path.name.split('.')[0])
    output_label_path = Path.joinpath(Path(output_path, "labels"), image_path.name.split('.')[0])
    # load image
    image = cv2.imread(image_path.as_posix())
    # load label file
    with label_path.open() as label_file:
        labels = label_file.readlines()

    # label: [x1, y1, x2, y2, class, ignore] origin for coordinates is top left corner of image
    for i, label in enumerate(labels):
        label_dict = convert_label_to_dict(label)
        box = label_dict['bbox']
        class_label = label_dict['class']
        # some boxes may extend past the image bounds
        confine_box(box, 0, 0, image.shape[0], image.shape[1])
        try:
            pad_to_square(box, 0, 0, image.shape[0], image.shape[1])
        except ValueError as exception:
            print('Warning: {} in image \'{}\''.format(exception, image_path))
            continue

        if box[2] - box[0] == 0 or box[3] - box[1] == 0:
            print('Warning: box {} with zero area detected in image \'{}\''.format(box, file_name))
            continue

        patch = image[box[1]:box[3], box[0]:box[2]]
        patch = cv2.resize(patch, (patch_size, patch_size))
        patch_file_path = '{}_{}.png'.format(output_image_path, i)
        cv2.imwrite(patch_file_path, patch)
        label_file_path = '{}_{}.txt'.format(output_label_path, i)
        with open(label_file_path, 'w') as label_file:
            label_file.write(str(class_label))


def create_output_folders(output_image_path, output_label_path):
    """Create output folders for label and patch data

    Arguments:
        output_image_path {str} -- Absolute path to the output folder for patch data
        output_label_path {str} -- Absolute path to the output folder for label data
    """
    if not output_image_path.is_dir():
        try:
            Path.mkdir(output_image_path)
        except OSError:
            print("Creation of the directory {} failed".format('patches'))
        else:
            print("Successfully created the directory {}".format('patches'))
    if not output_label_path.is_dir():
        try:
            Path.mkdir(output_label_path)
        except OSError:
            print("Creation of the directory {} failed".format('labels'))
        else:
            print("Successfully created the directory {}".format('labels'))


def convert_label_to_dict(label):
    """ Covnert label in string format into a dictionary

    Arguments:
        label {str} -- Label information for a single patch

    Returns:
        dict -- Label dictionary
    """
    label_dict = {}
    content = label.strip().split(' ')
    # Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', Misc'
    # or 'DontCare'
    label_dict['class'] = content[0]
    # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
    label_dict['truncated'] = float(content[1])
    # Integer(0, 1, 2, 3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded,
    # 3 = unknown
    label_dict['occluded'] = int(content[2])
    # Observation angle of object, ranging [-pi..pi]
    label_dict['alpha'] = float(content[3])
    # 2D bounding box of object in the image(0-based index): Contains left, top, right, bottom pixel coordinates
    label_dict['bbox'] = [round(float(x)) for x in content[4:8]]
    # 3D object dimensions: height, width, length(in meters)
    label_dict['dimensions'] = [float(x) for x in content[8:11]]
    # 3D object location x, y, z in camera coordinates(in meters)
    label_dict['location'] = [float(x) for x in content[11:14]]
    # Rotation ry around Y-axis in camera coordinates[-pi..pi]
    label_dict['rotation_y'] = float(content[14])
    return label_dict


def pad_to_square(box, ymin, xmin, ymax, xmax):
    """Pad the given box to a square shape.

    If the box resides near the edge of the given bounds, the padding may be shifted inwards.

    Arguments:
        box {list} -- box to pad
        ymin {int} -- minimum y value
        xmin {int} -- minimum x value
        ymax {int} -- maximum y value
        xmax {int} -- maximum x value

    Raises:
        ValueError: Cannot fit padded box in image
    """
    original = box.copy()
    height = box[3] - box[1]
    width = box[2] - box[0]
    padding = abs(width - height) / 2
    if width > height:
        box[1] -= math.ceil(padding)
        box[3] += math.floor(padding)
        if box[1] < ymin:
            box[3] += ymin - box[1]
            box[1] = ymin
        if box[3] > ymax:
            box[1] -= box[3] - ymax
            box[3] = ymax
    elif height > width:
        box[0] -= math.ceil(padding)
        box[2] += math.floor(padding)
        if box[0] < xmin:
            box[2] += xmin - box[0]
            box[0] = xmin
        if box[2] > xmax:
            box[0] -= box[2] - xmax
            box[2] = xmax
    if box[0] < xmin or box[2] > xmax or box[1] < ymin or box[3] > ymax:
        raise ValueError('Cannot fit padded box in image, tried {} for input {}'.format(box, original))


def confine_box(box, ymin, xmin, ymax, xmax):
    """Confine the box to the given bounds.

    2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates

    Arguments:
        box {[type]} -- box to adapt
        ymin {int} -- minimum y value
        xmin {int} -- minimum x value
        ymax {int} -- maximum y value
        xmax {int} -- maximum x value
    """
    box[0] = max(xmin, min(xmax, box[0]))  # width
    box[1] = max(ymin, min(ymax, box[1]))  # height
    box[2] = max(xmin, min(xmax, box[2]))  # width
    box[3] = max(ymin, min(ymax, box[3]))  # height


if __name__ == '__main__':
    main()
