// TorchExample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>



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
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("C://Github//DeepLearningStudy//trained_model//CharacterClassification.pt");
       // torch::from_file("",)
        std::cout << "test" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.msg() << std::endl << std::endl;
        std::cerr << e.backtrace() << std::endl << std::endl;
        return -1;
    }
}
