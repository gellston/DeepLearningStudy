
import os
import cv2
import numpy as np
import random


def augment_image(original_image, label_image):
    rows, cols = original_image.shape[:2]
    # Random flipping
    #rotation_value = random.randint(-1, 1)
    #aug_img_final = cv2.flip(original_image, rotation_value)
    #aug_label_final =  cv2.flip(label_image, rotation_value)


    # shifting

    tx = random.uniform(-0.35, 0.35) * cols
    ty = random.uniform(-0.35, 0.35) * rows
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    aug_img_final = cv2.warpAffine(original_image, M, (cols, rows))
    aug_label_final = cv2.warpAffine(label_image, M, (cols, rows))

    # cropROI
    x, y = int(max(tx, 0)), int(max(ty, 0))
    w, h = int((cols - abs(tx))), int((rows - abs(ty)))
    aug_img_final = aug_img_final[y:y + h, x:x + w]
    aug_img_final = cv2.resize(aug_img_final, (cols, rows))

    aug_label_final = aug_label_final[y:y + h, x:x + w]
    aug_label_final = cv2.resize(aug_label_final, (cols, rows))


    #blur_val = random.randint(2, 7)
    #aug_img_final = cv2.blur(aug_img_final, (blur_val, blur_val))

    # rotation
    Cx, Cy = rows, cols
    rand_angle = random.randint(-45, 45)
    M1 = cv2.getRotationMatrix2D((Cy // 2, Cx // 2), rand_angle, 1)
    M2 = cv2.getRotationMatrix2D((Cy // 2, Cx // 2), rand_angle, 1)
    aug_img_final = cv2.warpAffine(aug_img_final, M1, (cols, rows))
    aug_label_final = cv2.warpAffine(aug_label_final, M2, (cols, rows))

    return (aug_img_final , aug_label_final)


source_path = 'C:/Users/gellston/Desktop/PCB_Augmentation_Final_512_Rotation/'
target_path ='C:/Users/gellston/Desktop/augmentation_/'

filelist = sorted(os.listdir(source_path))
count = 40000
for filename in filelist:
    if filename == '.DS_Store': continue
    count = count + 1

    source_full_original_path = source_path + filename + '/source.jpg'
    source_full_label_path = source_path + filename + '/0.jpg'

    source_image = cv2.imread(source_full_original_path)
    source_label = cv2.imread(source_full_label_path)

    for index in range(0, 5):
        target_full_original_path1 = target_path + str(count) + "_" +  str(index) + '/'
        target_full_label_path1 = target_path + str(count) + "_" +  str(index) + '/'
        os.makedirs(target_full_label_path1, exist_ok=True)
        os.makedirs(target_full_original_path1, exist_ok=True)
        target_full_label_path1 = target_full_label_path1 + '0.jpg'
        target_full_original_path1 = target_full_original_path1 + 'source.jpg'

        aug_source, aug_label = augment_image(source_image, source_label)
        cv2.imwrite(target_full_original_path1, aug_source)
        cv2.imwrite(target_full_label_path1, aug_label)
