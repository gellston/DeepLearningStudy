import torch
import numpy as np

import torchvision.transforms.functional as ttf
import cv2
import math
import torchvision.transforms as transforms




## new
def gaussian_radius(det_size, min_overlap=0.7):
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    maxPixelValue = np.amax(gaussian)
    gaussian = gaussian / maxPixelValue

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def generate_heatmap(heatmap, center_x, center_y, bboxes_h, bboxes_w):

    radius = gaussian_radius((np.ceil(bboxes_h), np.ceil(bboxes_w)))
    radius = max(0, int(radius))

    draw_umich_gaussian(heatmap, (center_x, center_y), radius)

## new













def batch_loader(loader, batch_size, input_width, input_height, feature_map_scale, device):


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

        resized_color_image = ttf.resize(image, size=(input_width, input_height)) * 255
        image_list.append(resized_color_image)

        """
        ##opencv
        torch_image = resized_color_image.squeeze() * 255
        opencv_image = torch_image.numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        """

        size_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        offset_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        gaussian_map = np.zeros((int(feature_map_height), int(feature_map_width), 1), np.float32)

        bbox = label['bbox'][0]
        bbox_count = bbox.size(dim=0)
        for box_index in range(bbox_count):

            ##Fitting into input image size (Based on input image) Confirmed
            input_box_x = bbox[box_index][0] / color_image_width * input_width
            input_box_y = bbox[box_index][1] / color_image_height * input_height
            input_box_width = bbox[box_index][2] / color_image_width * input_width
            input_box_height = bbox[box_index][3] / color_image_height * input_height
            ##Fitting into input image size (Based on input image) Confirmed

            #red = (0, 0, 255)
            #cv2.rectangle(opencv_image, (int(input_box_x), int(input_box_y)),
            #             (int(input_box_x + input_box_width), int(input_box_y + input_box_height)), red, 3)

            ##Fitting into input image size (Based on feature image)
            feature_box_width = bbox[box_index][2] / color_image_width * feature_map_width
            feature_box_height = bbox[box_index][3] / color_image_height * feature_map_height
            feature_box_x = bbox[box_index][0] / color_image_width * feature_map_width + feature_box_width/2
            feature_box_y = bbox[box_index][1] / color_image_height * feature_map_height + feature_box_height/2

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
            generate_heatmap(gaussian_map[:, :, 0], clamp_feature_box_x, clamp_feature_box_y, feature_box_height, feature_box_width)
            ##Gaussian Map


        ##opencv
        """
        resized_gaussian_map = cv2.resize(gaussian_map[:, :, 0], dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
        cv2.imshow('gaussian map', resized_gaussian_map)
        cv2.imshow('rect visualizaiton', opencv_image)
        cv2.waitKey(100)
        """


        gaussian_map_tensor = tensorTransform(gaussian_map).unsqueeze(dim=0)
        size_map_list.append(size_map)
        offset_map_list.append(offset_map)
        gaussian_map_list.append(gaussian_map_tensor)

        temp_batch_size = temp_batch_size-1
        if temp_batch_size == 0:
            break

    image_batch = torch.cat(image_list, dim=0).to(device)
    gaussian_map_batch = torch.cat(gaussian_map_list, dim=0).to(device)
    size_map_batch = torch.cat(size_map_list, dim=0).to(device)
    offset_map_batch = torch.cat(offset_map_list, dim=0).to(device)

    return (image_batch, gaussian_map_batch, size_map_batch, offset_map_batch)
