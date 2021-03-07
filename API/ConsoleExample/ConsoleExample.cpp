// ConsoleExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "deep.h"


#include <opencv2/opencv.hpp>


int main()
{

    hv::v1::deep::segmentation segmentation;

    segmentation.import("C://Github//DeepLearningStudy//trained_model//SelfieSegmentation//");


    cv::Mat original = cv::imread("C://Github//DeepLearningStudy//test_image//00001.jpg");
    cv::Mat resized_input_image = cv::Mat(256, 256, CV_8UC1);
    cv::Mat resized_output_probability = cv::Mat(256, 256, CV_32FC1);
    cv::Mat resized_output_threshold = cv::Mat(cv::Size(256, 256), CV_32FC1);

    cv::resize(original, resized_input_image, cv::Size(256, 256));


    auto output = segmentation.run(resized_input_image.data, 256, 256, 3, 1);

    memcpy(resized_output_probability.data, output.get(), 256 * 256 * 1 * sizeof(float));

    resized_output_probability = resized_output_probability * 255;

    cv::threshold(resized_output_probability, resized_output_threshold, 210, 255, cv::THRESH_BINARY);


    cv::imshow("original", original);
    cv::imshow("original_resized", resized_input_image);
    cv::imshow("resized_output_threshold", resized_output_threshold);
    cv::imshow("resized_output_probability", resized_output_probability);

    cv::waitKey();

    return 0;
}
