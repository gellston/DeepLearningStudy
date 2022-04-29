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






def batch_loader(loader, batch_size, input_width, input_height, feature_map_scale, device, isNorm=True):

    image_list = []
    size_map_list = []
    offset_map_list = []
    gaussian_map_list = []
    resized_bbox_list = []
    image_size_list = []

    feature_map_width = input_width / feature_map_scale
    feature_map_height = input_height / feature_map_scale

    tensorTransform = transforms.ToTensor()
    temp_batch_size = 0

    #box counting for normalization
    box_count = 0
    for image, label in loader:
        color_image = image[0]
        color_image_width = color_image.size(dim=2)
        color_image_height = color_image.size(dim=1)

        resized_color_image = ttf.resize(image, size=(input_width, input_height))

        if isNorm == False:
            resized_color_image = resized_color_image * 255

        image_list.append(resized_color_image)
        image_size_list.append((color_image_width, color_image_height))

        size_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        offset_map = torch.zeros([1, 2, int(feature_map_height), int(feature_map_width)], dtype=torch.float)
        gaussian_map = np.zeros((int(feature_map_height), int(feature_map_width), 1), dtype=np.float32)

        bbox = label['bbox'][0]
        bbox_count = bbox.size(dim=0)
        bbox_list = []
        for box_index in range(bbox_count):
            ##Fitting into input image size (Based on feature image)
            feature_box_width = bbox[box_index][2].item() / color_image_width * feature_map_width
            feature_box_height = bbox[box_index][3].item() / color_image_height * feature_map_height
            feature_box_x = bbox[box_index][0].item() / color_image_width * feature_map_width + feature_box_width/2
            feature_box_y = bbox[box_index][1].item() / color_image_height * feature_map_height + feature_box_height/2

            input_image_box_width = bbox[box_index][2].item() / color_image_width * input_width
            input_image_box_height = bbox[box_index][3].item() / color_image_height * input_height
            input_image_box_x = bbox[box_index][0].item() / color_image_width * input_width
            input_image_box_y = bbox[box_index][1].item() / color_image_height * input_height

            bbox_list.append((input_image_box_x, input_image_box_y, input_image_box_width, input_image_box_height))

            ##Clamping x,y
            clamp_feature_box_x = np.clip(feature_box_x, 0, feature_map_width-1)
            clamp_feature_box_y = np.clip(feature_box_y, 0, feature_map_height-1)
            clamp_feature_box_x = int(clamp_feature_box_x)
            clamp_feature_box_y = int(clamp_feature_box_y)


            ##Calculating offset x, y
            feature_box_offset_x = feature_box_x - clamp_feature_box_x
            feature_box_offset_y = feature_box_y - clamp_feature_box_y

            ##Fill Size Map
            size_map[0][0][clamp_feature_box_y][clamp_feature_box_x] = feature_box_width
            size_map[0][1][clamp_feature_box_y][clamp_feature_box_x] = feature_box_height

            ##Fill Offset Map
            offset_map[0][0][clamp_feature_box_y][clamp_feature_box_x] = feature_box_offset_x
            offset_map[0][1][clamp_feature_box_y][clamp_feature_box_x] = feature_box_offset_y

            ##Gaussian Map
            generate_heatmap(gaussian_map[:, :, 0], clamp_feature_box_x, clamp_feature_box_y, feature_box_height, feature_box_width)
            ##Gaussian Map

        resized_bbox_list.append(bbox_list)

        gaussian_map_tensor = tensorTransform(gaussian_map).unsqueeze(dim=0)
        size_map_list.append(size_map)
        offset_map_list.append(offset_map)
        gaussian_map_list.append(gaussian_map_tensor)

        temp_batch_size = temp_batch_size+1
        if temp_batch_size == batch_size:
            break

    image_batch = torch.cat(image_list, dim=0).to(device)
    gaussian_map_batch = torch.cat(gaussian_map_list, dim=0).to(device)
    size_map_batch = torch.cat(size_map_list, dim=0).to(device)
    offset_map_batch = torch.cat(offset_map_list, dim=0).to(device)

    return (image_batch, resized_bbox_list, gaussian_map_batch, size_map_batch, offset_map_batch)


def batch_accuracy(input_image_width,
                   input_image_height,
                   scale_factor,
                   score_threshold,
                   iou_threshold,
                   gaussian_map_batch,
                   size_map_batch,
                   offset_map_batch,
                   bbox_list):

    feature_image_width = int(input_image_width / scale_factor)
    feature_image_height = int(input_image_height / scale_factor)
    average_accuracy = 0

    prediction_box_batch_result_list = []
    prediction_box_batch_list = []

    batch_size = gaussian_map_batch.size(0)
    for batch_index in range(batch_size):
        prediction_box_list = []
        gaussian_map = gaussian_map_batch[batch_index]
        size_map = size_map_batch[batch_index]
        offset_map = offset_map_batch[batch_index]

        ## Box extraction from feature map
        for feature_y in range(feature_image_height):
            for feature_x in range(feature_image_width):
                if gaussian_map[0, feature_y, feature_x].item() > score_threshold:
                    offset_x = offset_map[0, feature_y, feature_x].item()
                    offset_y = offset_map[1, feature_y, feature_x].item()
                    final_box_width = size_map[0, feature_y, feature_x].item() / feature_image_width * input_image_width
                    final_box_height = size_map[1, feature_y, feature_x].item() / feature_image_height * input_image_height
                    final_box_x = (feature_x + offset_x) / feature_image_width * input_image_width - final_box_width / 2
                    final_box_y = (feature_y + offset_y) / feature_image_height * input_image_height - final_box_height / 2

                    probability = gaussian_map[0, feature_y, feature_x].item()
                    if final_box_width != 0 and final_box_height != 0:
                        prediction_box_list.append((probability, final_box_x, final_box_y, final_box_width, final_box_height))

        prediction_box_batch_list.append(prediction_box_list)

    prediction_box_batch_length = len(prediction_box_batch_list)
    if prediction_box_batch_length == 0:
        return

    for batch_index in range(batch_size):
        prediction_box_list = []
        one_batch_box_count = len(bbox_list[batch_index])
        for bbox in bbox_list[batch_index]:
            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_width = bbox[2]
            bbox_height = bbox[3]

            for prediction_box in prediction_box_batch_list[batch_index]:
                prediction_probability = prediction_box[0]
                prediction_box_x = prediction_box[1]
                prediction_box_y = prediction_box[2]
                prediction_box_width = prediction_box[3]
                prediction_box_height = prediction_box[4]
                if BOX_IOU(bbox_x, bbox_y, bbox_width, bbox_height, prediction_box_x, prediction_box_y,
                           prediction_box_width, prediction_box_height) > iou_threshold:
                    average_accuracy += (1 / one_batch_box_count / batch_size)
                    prediction_box_list.append((prediction_probability, prediction_box_x, prediction_box_y,
                                                prediction_box_width, prediction_box_height))
                    break;

        prediction_box_batch_result_list.append(prediction_box_list)

    return (average_accuracy, prediction_box_batch_result_list)


def batch_prediction_loader(loader, batch_size, input_width, input_height, device, isNorm=True):
    image_list = []
    temp_batch_size = 0
    for image, label in loader:
        resized_color_image = ttf.resize(image, size=(input_width, input_height))
        if isNorm == False:
            resized_color_image = resized_color_image * 255
        image_list.append(resized_color_image)
        temp_batch_size = temp_batch_size+1
        if temp_batch_size == batch_size:
            break
    image_batch = torch.cat(image_list, dim=0).to(device)
    return image_batch


def batch_box_extractor(input_image_width,
                       input_image_height,
                       scale_factor,
                       score_threshold,
                       gaussian_map_batch,
                       size_map_batch,
                       offset_map_batch):

    feature_image_width = int(input_image_width / scale_factor)
    feature_image_height = int(input_image_height / scale_factor)
    prediction_box_batch_list = []

    batch_size = gaussian_map_batch.size(0)
    for batch_index in range(batch_size):
        prediction_box_list = []
        gaussian_map = gaussian_map_batch[batch_index]
        size_map = size_map_batch[batch_index]
        offset_map = offset_map_batch[batch_index]

        ## Box extraction from feature map
        for feature_y in range(feature_image_height):
            for feature_x in range(feature_image_width):
                if gaussian_map[0, feature_y, feature_x].item() > score_threshold:
                    offset_x = offset_map[0, feature_y, feature_x].item()
                    offset_y = offset_map[1, feature_y, feature_x].item()
                    final_box_width = size_map[0, feature_y, feature_x].item() / feature_image_width * input_image_width
                    final_box_height = size_map[1, feature_y, feature_x].item() / feature_image_height * input_image_height
                    final_box_x = (feature_x + offset_x) / feature_image_width * input_image_width - final_box_width / 2
                    final_box_y = (feature_y + offset_y) / feature_image_height * input_image_height - final_box_height / 2

                    if final_box_width != 0 and final_box_height != 0:
                        prediction_box_list.append((final_box_x, final_box_y, final_box_width, final_box_height))

        prediction_box_batch_list.append(prediction_box_list)

    prediction_box_batch_length = len(prediction_box_batch_list)
    if prediction_box_batch_length == 0:
        return

    return prediction_box_batch_list

def BOX_IOU(boxA_x, boxA_y, boxA_width, boxA_height,
            boxB_x, boxB_y, boxB_width, boxB_height):

    center_cordinate_boxA_sx = boxA_x
    center_cordinate_boxA_sy = boxA_y
    center_cordinate_boxA_ex = boxA_x + boxA_width
    center_cordinate_boxA_ey = boxA_y + boxA_height

    center_cordinate_boxB_sx = boxB_x
    center_cordinate_boxB_sy = boxB_y
    center_cordinate_boxB_ex = boxB_x + boxB_width
    center_cordinate_boxB_ey = boxB_y + boxB_height

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(center_cordinate_boxA_sx, center_cordinate_boxB_sx)    #SX
    y1 = max(center_cordinate_boxA_sy, center_cordinate_boxB_sy)    #SY
    x2 = min(center_cordinate_boxA_ex, center_cordinate_boxB_ex)    #EX
    y2 = min(center_cordinate_boxA_ey, center_cordinate_boxB_ey)    #EY

    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (center_cordinate_boxA_ex - center_cordinate_boxA_sx + 1) * (center_cordinate_boxA_ey - center_cordinate_boxA_sy + 1)
    boxBArea = (center_cordinate_boxB_ex - center_cordinate_boxB_sx + 1) * (center_cordinate_boxB_ey - center_cordinate_boxB_sy + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value

    return iou

