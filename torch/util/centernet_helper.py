import torch
import numpy as np

import torchvision.transforms.functional as ttf
import cv2
import math
import torchvision.transforms as transforms




## new
def gaussian_radius_2(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian_radius(det_size, min_overlap=0.7):
    det_h, det_w = det_size
    rh = 0.1155 * det_h
    rw = 0.1155 * det_w
    return rh, rw


def gaussian2D(shape, sigma_w=1, sigma_h=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_w * sigma_w) + (y * y) / (2 * sigma_h * sigma_h)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian_2(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_gaussian(heatmap, center, radius_h, radius_w, k=1):
    diameter_h = 2 * radius_h + 1
    diameter_w = 2 * radius_w + 1
    gaussian = gaussian2D((diameter_h, diameter_w), sigma_w=diameter_w / 6, sigma_h=diameter_h / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_w), min(width - x, radius_w + 1)
    top, bottom = min(y, radius_h), min(height - y, radius_h + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_h - top:radius_h + bottom, radius_w - left:radius_w + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

## new




def batch_loader(loader, batch_size, input_width, input_height, feature_map_scale):


    image_list = []
    size_map_list = []
    offset_map_list = []
    gaussian_map_list = []

    feature_map_width = input_width / feature_map_scale
    feature_map_height = input_height / feature_map_scale


    tensorTransform = transforms.ToTensor()

    temp_batch_size = batch_size

    for image, label in loader:

        color_image = image[0]
        color_image_width = color_image.size(dim=2)
        color_image_height = color_image.size(dim=1)

        resized_color_image = ttf.resize(image, size=(input_width, input_height))
        image_list.append(resized_color_image)

        size_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        offset_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        gaussian_map = np.zeros((int(feature_map_height), int(feature_map_width), 1), np.float32)

        bbox = label['bbox'][0]
        bbox_count = bbox.size(dim=0)
        for box_index in range(bbox_count):

            ##Fitting into input image size (Based on input image)
            input_box_width = bbox[box_index][2] / color_image_width * input_width
            input_box_height = bbox[box_index][3] / color_image_height * input_height

            ##Fitting into input image size (Based on feature image)
            feature_box_width = bbox[box_index][2] / feature_map_width * input_width
            feature_box_height = bbox[box_index][3] / feature_map_height * input_height
            feature_box_x = bbox[box_index][0] / feature_map_width * input_width + feature_box_width/2
            feature_box_y = bbox[box_index][1] / feature_map_height * input_height + feature_box_height/2

            ##Clamping x,y
            clamp_feature_box_x = np.clip(feature_box_x, 0, feature_map_width)
            clamp_feature_box_y = np.clip(feature_box_y, 0, feature_map_height)

            ##Calculating offset x, y
            feature_box_offset_x = feature_box_x - clamp_feature_box_x
            feature_box_offset_y = feature_box_y - clamp_feature_box_y

            ##Fill Size Map
            size_map[0][0][int(clamp_feature_box_y-1)][int(clamp_feature_box_x-1)] = input_box_width
            size_map[0][1][int(clamp_feature_box_y-1)][int(clamp_feature_box_x-1)] = input_box_height

            ##Fill Offset Map
            offset_map[0][0][int(clamp_feature_box_y-1)][int(clamp_feature_box_x-1)] = feature_box_offset_x
            offset_map[0][1][int(clamp_feature_box_y-1)][int(clamp_feature_box_x-1)] = feature_box_offset_y

            ##Gaussian Map
            radius_h, radius_w = gaussian_radius((math.ceil(feature_box_height), math.ceil(feature_box_width)))
            radius_h = max(0, int(radius_h))
            radius_w = max(0, int(radius_w))

            radius = gaussian_radius_2((math.ceil(feature_box_height), math.ceil(feature_box_width)))
            radius = max(0, int(radius))
            ct = np.array([int(feature_box_x), int(feature_box_y)], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(gaussian_map[:, :, 0], ct_int, radius_h, radius_w)
            draw_gaussian_2(gaussian_map[:, :, 0], ct_int, radius)


        gaussian_map_tensor = tensorTransform(gaussian_map)
        size_map_list.append(size_map)
        offset_map_list.append(offset_map)
        gaussian_map_list.append(gaussian_map_tensor)

        temp_batch_size = temp_batch_size-1
        if temp_batch_size == 0:
            break

    image_batch = torch.cat(image_list, dim=0)
    gaussian_map_batch = torch.cat(gaussian_map_list, dim=0)
    size_map_batch = torch.cat(size_map_list, dim=0)
    offset_map_batch = torch.cat(offset_map_list, dim=0)

    return (image_batch, gaussian_map_batch, size_map_batch, offset_map_batch)
