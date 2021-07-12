// ConsoleExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "deep.h"


#include <opencv2/opencv.hpp>


int main()
{

    hv::v1::deep::segmentation segmentation;

    segmentation.import("C://Github//DeepLearningStudy//trained_model//OnyxSegmentation//");


    cv::Mat original = cv::imread("C://Users//gellston//Desktop//OnyxAugmentation//1_202106291149473//source.jpg",cv::IMREAD_GRAYSCALE);
    cv::Mat resized_input_image = cv::Mat(512, 512, CV_8UC1);
    cv::Mat resized_output_probability = cv::Mat(512, 512, CV_32FC1);
    cv::Mat resized_output_threshold = cv::Mat(cv::Size(512, 512), CV_32FC1);

    cv::resize(original, resized_input_image, cv::Size(512, 512));


    segmentation.run(resized_input_image.data, resized_output_probability.data, 512, 512, 1, 1);

   // memcpy(resized_output_probability.data, output.get(), 512 * 512 * 1 * sizeof(float));

    resized_output_probability = resized_output_probability * 512;

    cv::threshold(resized_output_probability, resized_output_threshold, 240, 255, cv::THRESH_BINARY);


    cv::imshow("original", original);
    cv::imshow("original_resized", resized_input_image);
    cv::imshow("resized_output_threshold", resized_output_threshold);
    cv::imshow("resized_output_probability", resized_output_probability);

    cv::waitKey();

    return 0;
}
