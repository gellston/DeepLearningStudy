// ConsoleExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "deep.h"


#include <opencv2/opencv.hpp>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

int main()
{

    hv::v1::deep::segmentation segmentation;

    segmentation.import("C://Github//DeepLearningStudy//trained_model//ModuleExample//");






 



    while (true) {
        int count = 0;
        float average_cost = 0;
        float average_accuracy = 0;
        for (int index = 1; index < 235; index++) {
            count++;
            //std::cout << "current index = " << index << std::endl;
            if (index == 38 || index == 41 || 
                index == 48 || index == 55 || 
                index == 56 || index == 77 || 
                index == 95 || index == 111 || index == 114 || index == 115 || index == 126
                || index == 156 || index == 222 || index == 227 || index == 233 || index == 235) continue;
            std::string original_path = "C://Github//DeepLearningStudy//dataset//portrait_segmentation_input256x256//";
            std::string label_path = "C://Github//DeepLearningStudy//dataset//portrait_segmentation_label256x256//";

            char number[25];
            sprintf(number, "%05d", index);

            original_path += number;
            original_path += ".jpg";

            label_path += number;
            label_path += ".jpg";

            cv::Mat original = cv::imread(original_path, cv::IMREAD_COLOR);
            cv::Mat label = cv::imread(label_path, cv::IMREAD_GRAYSCALE);
            cv::Mat resized_original_probability = cv::Mat(256, 256, CV_32FC3);
            cv::Mat resized_label_probability = cv::Mat(256, 256, CV_32FC1);
            cv::Mat resized_label_threshold = cv::Mat(256, 256, CV_32FC1);



            cv::Mat resized_output_probability = cv::Mat(256, 256, CV_32FC1);


            original.convertTo(resized_original_probability, CV_32FC3);
            label.convertTo(resized_label_probability, CV_32FC1);
            cv::threshold(resized_label_probability, resized_label_threshold, 128, 1, cv::ThresholdTypes::THRESH_BINARY);

            //resized_label_threshold = resized_label_threshold / 255;


            auto cost = segmentation.train((float*)resized_original_probability.data, 256, 256, 3, (float*)resized_label_threshold.data, 256, 256, 1, 1);
            auto accuracy = segmentation.accuracy((float*)resized_original_probability.data, 256, 256, 3, (float*)resized_label_threshold.data, 256, 256, 1, 1);
            segmentation.run((float*)resized_original_probability.data, (float*)resized_output_probability.data, 256, 256, 3, 1);
            cv::threshold(resized_output_probability, resized_output_probability, 0.6, 1, cv::ThresholdTypes::THRESH_BINARY);
            resized_output_probability = resized_output_probability * 255;
            average_cost += cost;
            average_accuracy += accuracy;

            cv::imshow("original", original);
            cv::imshow("label probability", resized_label_probability);
            cv::imshow("label threshold", resized_label_threshold);
            cv::imshow("output", resized_output_probability);
            cv::waitKey(10);

        }
        std::cout << "cost : " << std::to_string(average_cost / count) << std::endl;
        std::cout << "accuracy : " << std::to_string(average_accuracy / count) << std::endl;
    }
    

   // memcpy(resized_output_probability.data, output.get(), 512 * 512 * 1 * sizeof(float));




    return 0;
}
