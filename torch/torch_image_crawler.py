import os
import cv2 as cv2
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from datetime import date


# crawer setting
folder_name = 'C://Github//Crawling//'
start_year = 2011
period = 10
image_width = 224
image_height = 224



classCount = 0

for className in os.listdir(folder_name):

    if className == '.DS_Store': continue

    classCount = 0
    fullPath = folder_name + "/" + className;




    for year in range(period):
        filters = dict(date=((year + start_year, 1, 1), (year + start_year, 12, 24)))
        google_crawler = GoogleImageCrawler(storage={'root_dir': folder_name + className + '/'},
                                            feeder_threads=1,
                                            parser_threads=1,
                                            downloader_threads=4)
        google_crawler.crawl(keyword=className,
                             filters=filters,
                             max_num=2000,
                             file_idx_offset='auto',
                             max_size=None)

    classCount = classCount + 1
    fileIndex = 0

    if os.path.isdir(fullPath):
        fileList = os.listdir(fullPath)
        for fileName in fileList:
            if fileName == '.DS_Store' : continue

            filePath = fullPath + "/" + fileName
            filename, file_extension = os.path.splitext(filePath)
            if file_extension != '.jpg' and file_extension != '.bmp' and file_extension != '.gif' and file_extension != '.jpeg' and file_extension != '.png':
                os.remove(filePath)
                continue

            fileIndex = fileIndex + 1
            modifiedPath = fullPath + "/" + str(fileIndex) + "_" + str(classCount) + "_" + className + '.jpg'
            original = cv2.imread(filePath)

            if original is None:
                os.remove(filePath)
                fileIndex = fileIndex = fileIndex - 1
                continue

            if len(original.shape) != 3:
                os.remove(filePath)
                fileIndex = fileIndex = fileIndex - 1
                continue

            resizeImage = cv2.resize(original, (image_width, image_height), interpolation=cv2.INTER_AREA)
            npImage = []

            cv2.imwrite(modifiedPath, resizeImage)
            cv2.imshow('resize', resizeImage)
            cv2.waitKey(10)
            os.remove(filePath)