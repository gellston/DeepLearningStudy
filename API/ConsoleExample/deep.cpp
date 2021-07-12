#include "deep.h"


#include <iostream>
#include <array>

static void deallocator(void* data, size_t length, void* arg) {

}


hv::v1::deep::segmentation::segmentation() : run_options(nullptr),
										     session_options(nullptr),
											 graph(nullptr),
											 status(nullptr),
											 session(nullptr),
										     is_loaded(false){

	this->run_options = TF_NewBufferFromString("", 0);
	this->session_options = TF_NewSessionOptions();
	this->graph = TF_NewGraph();
	this->status = TF_NewStatus();

	hv::v1::raii* raii = new hv::v1::raii([&] {

		if (this->run_options != nullptr)
			TF_DeleteBuffer(this->run_options);


		if (this->session_options != nullptr)
			TF_DeleteSessionOptions(this->session_options);


		if (this->graph != nullptr)
			TF_DeleteGraph(this->graph);


		if (this->status != nullptr)
			TF_DeleteStatus(this->status);


		if (this->session != nullptr && this->status != nullptr)
			TF_DeleteSession(this->session, status);

	});

	this->raii_destructor = std::shared_ptr<hv::v1::raii>(raii);

}


hv::v1::deep::segmentation::~segmentation() {

}

void hv::v1::deep::segmentation::import(std::string path) {

	if (this->is_loaded == true) return;



	std::array<char const*, 1> tags = { "serve" };

	this->session = TF_LoadSessionFromSavedModel(session_options, run_options, path.c_str(), tags.data(), tags.size(), graph, nullptr, status);

	if (TF_GetCode(status) != TF_OK) {
		std::cout << TF_Message(status) << '\n';
		this->is_loaded = false;
		return;
	}



	//this->input_op = TF_GraphOperationByName(this->graph, "serving_default_x_input_node:0");
	//if (input_op == nullptr) {
	//	std::cout << "Failed to find graph operation\n";
	//	this->is_loaded = false;
	//	return;
	//}

	//this->output_op = TF_GraphOperationByName(this->graph, "StatefulPartitionedCall:0");
	//if (output_op == nullptr) {
	//	std::cout << "Failed to find graph operation\n";
	//	this->is_loaded = false;
	//	return;
	//}


	this->is_loaded = true;

}

void hv::v1::deep::segmentation::run(unsigned char* input_buffer, unsigned char * output_buffer, int width, int height, int channel, int label) {

	//std::shared_ptr<float> output((float*)malloc(sizeof(float) * width * height * label));
//
	///memset(output.get(), 0, sizeof(float) * width * height * label);

	std::vector<TF_Output> input_vec;
	std::vector<TF_Output> output_vec;


	std::vector<TF_Tensor*> input_tensors;
	std::vector<TF_Tensor*> output_tensors(1);

	hv::v1::raii input_destructor([&] {
		for (TF_Tensor* tensor : input_tensors) {
			TF_DeleteTensor(tensor);
		}
	});

	hv::v1::raii output_destructor([&] {
		for (TF_Tensor* tensor : output_tensors) {
			TF_DeleteTensor(tensor);
		}
	});


	input_vec.push_back({ TF_GraphOperationByName(this->graph, "serving_default_x_input_node"), 0 });
	output_vec.push_back({ TF_GraphOperationByName(this->graph, "StatefulPartitionedCall"), 0 });


	unsigned int input_size = width * height * channel;
	std::vector<float> input_data;
	input_data.resize(input_size);
	float* input_data_ptr = input_data.data();

	for (int index = 0; index < input_size; index++) {
		input_data_ptr[index] = input_buffer[index];
	}

	int64_t in_dims[] = { 1, height, width, channel };



	//auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII

	input_tensors.push_back(TF_NewTensor(TF_FLOAT, in_dims, 4, input_data_ptr, input_size*sizeof(float), deallocator, 0));


	try {
		TF_SessionRun(this->session,
			this->run_options,
			input_vec.data(), input_tensors.data(), input_tensors.size(),
			output_vec.data(), output_tensors.data(), output_vec.size(),
			nullptr, 0,
			nullptr,
			status);


		if (TF_GetCode(this->status) != TF_OK) {
			throw std::exception("invalid status");
		}

		const auto data = static_cast<float*>(TF_TensorData(output_tensors[0]));
		memcpy(output_buffer, data, sizeof(float) * width * height * label);

		//return output;
		return;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		throw std::exception("run model failed.");
	}




	throw std::exception("run model failed.");
}