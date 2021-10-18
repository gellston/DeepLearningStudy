// TorchExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>



#include <opencv2/opencv.hpp>



#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>

int main()
{
    torch::DeviceType device;
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
    }
    else
    {
        device = torch::kCPU;
    }
    torch::Device aTorchDevice = torch::Device(device);


    torch::jit::script::Module module;
    try {
        
        
        module = torch::jit::load("C://Github//DeepLearningStudy//trained_model//NoTrainCharacterClassification.pt");
        module.to(device);
       // torch::from_file("",)
        std::cout << "test" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.msg() << std::endl << std::endl;
        std::cerr << e.backtrace() << std::endl << std::endl;
        return -1;
    }

    
    // Inference Test
    std::string inputPath = "C://Github//DeepLearningStudy//dataset//digits-train//0_zero//1_1_0_zero.jpg";
    cv::Mat input = cv::imread(inputPath, cv::ImreadModes::IMREAD_GRAYSCALE);

    std::vector<torch::jit::IValue> tensor_vec;
    for (int index = 0; index < 1; index++) {
        torch::Tensor tensor_image = torch::from_blob(input.data, {input.rows * input.cols }, torch::kByte).to(torch::kFloat32).cuda();
        tensor_vec.push_back(tensor_image);

    }

    try {
        module.eval();
        auto output = module.forward(tensor_vec).toTensor().cpu();
        std::vector<float> vector_output(10);
        std::memcpy(vector_output.data(), output.data_ptr(), sizeof(float) * 10);
        for (int index = 0; index < 9; index++)
            std::cout << "result : [" << index << "]: accuracy:" << vector_output[index] << std::endl;

    }
    catch (const c10::Error& e) {
        std::cerr << e.msg() << std::endl << std::endl;
        std::cerr << e.backtrace() << std::endl << std::endl;
        return -1;
    }
    // Inference Test




    // Training Test

    // Training Data file names 

    std::vector<std::string> file_names;
    for (std::filesystem::recursive_directory_iterator i("C://Github//DeepLearningStudy//dataset//digits-train"), end; i != end; ++i) {
        if (!is_directory(i->path())) {
            //std::cout << i->path() << std::endl;
            file_names.push_back(i->path().u8string());

        }
    }
    auto rng = std::default_random_engine{};
    // Shuffle filenames
    std::shuffle(std::begin(file_names), std::end(file_names), rng);



    std::vector<std::string> label_info({ "0_zero","1_one","2_two","3_three","4_four","5_five","6_six","7_seven","8_eight","9_nine"});


    int batch_size = 10;
    int total_batches = file_names.size() / batch_size;
    double learning_rate = 0.01;

    // Training Get Parameters
    std::vector<torch::Tensor> parameters;
    for (const auto& params : module.parameters()) {
        parameters.push_back(params);
    }

    // Ready optimizer , Cross entropy function
   
    torch::optim::SGD optimizer(parameters, learning_rate);
    torch::nn::CrossEntropyLoss cost_function;

    // Epoch Loop Start

    try {
        for (int epoch = 0; epoch < 200; epoch++) {

            double average_cost = 0;
            double average_accuracy = 0;
            std::cout << "current epoch = " << epoch << std::endl;



            // Batch Loop Start
            for (int current_batch = 0; current_batch < total_batches; current_batch++) {
                module.train();
                optimizer.zero_grad(); // optimzier reset
                // Batch Making
                std::vector<std::string> one_batch;
                for (int index =  current_batch * batch_size; index < current_batch * batch_size + batch_size; index++) {
                    one_batch.push_back(file_names[index]);
                }

                std::vector<torch::jit::IValue> train_input;
                std::vector<torch::Tensor> vector_train_input;
                torch::Tensor tensor_label = torch::zeros({ batch_size}, torch::kLong).cuda();
                for (int index = 0; index < batch_size; index++) {
                    cv::Mat input = cv::imread(one_batch[index], cv::ImreadModes::IMREAD_GRAYSCALE);
                    torch::Tensor tensor_image = torch::from_blob(input.data, { input.rows * input.cols }, torch::kByte).to(torch::kFloat32);
                    vector_train_input.push_back(tensor_image);

                    for (int label_index = 0; label_index < 10; label_index++) {
                        if (one_batch[index].find(label_info[label_index]) != std::string::npos) {
                            tensor_label[index] = label_index;
                        }
                    }
                }
                auto stacked_input = torch::stack(vector_train_input).cuda();
                // std::cout << stacked_input.sizes() << std::endl;
                // std::cout << tensor_label << std::endl;
                train_input.push_back(stacked_input);

                // Batch Making

                auto output = module.forward(train_input).toTensor().cuda();
                torch::Tensor loss = cost_function(output, tensor_label);
                average_cost += (loss.item<float>() / total_batches);
                loss.backward();
                optimizer.step();



                module.eval();
                output = module.forward(train_input).toTensor().cuda();
                auto pred = output.argmax(1);
                average_accuracy += ((double)pred.eq(tensor_label).sum().item<int64_t>() / (double)batch_size);
            }

            average_accuracy = (average_accuracy / (double)total_batches);

            std::cout << "average cost = " << average_cost << std::endl;
            std::cout << "average accuracy " << average_accuracy << std::endl;


            if (average_accuracy > 0.95) {
                module.train();
                module.save("C://Github//DeepLearningStudy//trained_model//CharacterClassificationFromCpp.pt");
                break;
            }
        }
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return -1;
    }
    

    // Training Test



}
