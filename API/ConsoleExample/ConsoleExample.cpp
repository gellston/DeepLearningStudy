// ConsoleExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <deep.h>


#include <opencv2/opencv.hpp>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <Windows.h>
#include <profileapi.h>

int main()
{

    LARGE_INTEGER Frequency;
    LARGE_INTEGER BeginTime;
    LARGE_INTEGER Endtime;
    __int64 elapsed;
    double duringtime;


    hv::v1::deep::segmentation segmentation;

    segmentation.import("C://Github//DeepLearningStudy//trained_model//PCBDefectSegmentation//");


    std::string original_path = "C://Github//DeepLearningStudy//dataset//PCB_Augmentation_Final//0_20210727204810163//source.jpg";


    cv::Mat original = cv::imread(original_path, cv::IMREAD_COLOR);
    cv::Mat resized_original;

    cv::resize(original, resized_original, cv::Size(512, 512));
    cv::Mat resized_label_probability = cv::Mat(512, 512, CV_32FC1);
    cv::Mat grayScale = cv::Mat(512, 512, CV_8UC1);

    //프로그램이나 클래스 시작부분에
    QueryPerformanceFrequency(&Frequency);

    //사용하고자 하는 부분에 다음 코딩
    QueryPerformanceCounter(&BeginTime);

    int fps = 0;
    while (true) {

        //cv::Mat resized_original_probability = cv::Mat(512, 512, CV_32FC3);
       
        //cv::Mat resized_label_threshold = cv::Mat(512, 512, CV_32FC1);


        segmentation.run(resized_original.data, resized_label_probability.data, 512, 512, 3, 1);
        cv::Mat output = resized_label_probability * 255;
        cv::threshold(resized_label_probability, output, 128, 255, cv::ThresholdTypes::THRESH_BINARY);
        
        output.convertTo(grayScale, CV_8UC1);

        cv::imshow("result", resized_label_probability);
        cv::imshow("original", resized_original);
        cv::waitKey(10);



        QueryPerformanceCounter(&Endtime);
        elapsed = Endtime.QuadPart - BeginTime.QuadPart;
        duringtime = (double)elapsed / (double)Frequency.QuadPart;

        duringtime *= 1000;

        if (duringtime > 1000) {
            duringtime = 0;

            //사용하고자 하는 부분에 다음 코딩
            QueryPerformanceCounter(&BeginTime);

            std::cout << "performance fps = " << std::to_string(fps) << std::endl;
            fps = 0;

        }

        fps++;
    }
    


    return 0;
}
