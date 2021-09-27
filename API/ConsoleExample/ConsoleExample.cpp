// ConsoleExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <classification.h>


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


    deep::classification classification;

    classification.import("C://Github//DeepLearningStudy//trained_model//LeafletLineClassification//");


    std::string original_path = "C://Github//LeafletImageCropTool//Data//ClassificationExperiment//0_pass//2_1.jpg";


    cv::Mat original = cv::imread(original_path, cv::IMREAD_COLOR);



    //프로그램이나 클래스 시작부분에
    QueryPerformanceFrequency(&Frequency);

    //사용하고자 하는 부분에 다음 코딩
    QueryPerformanceCounter(&BeginTime);

    int fps = 0;
    float output[2];

    
    while (true) {

        //cv::Mat resized_original_probability = cv::Mat(512, 512, CV_32FC3);
       
        //cv::Mat resized_label_threshold = cv::Mat(512, 512, CV_32FC1);

        memset(output, 0, sizeof(float) * 2);

        classification.run(original.data, output, 512, 100, 3, 2);
     



        //cv::imshow("original", original);
        //cv::waitKey(10);



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
