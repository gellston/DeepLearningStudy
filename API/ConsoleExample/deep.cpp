#include "deep.h"
#include "raii.h"

#include <iostream>
#include <array>

#include <tensorflow/c/c_api.h>


namespace hv::v1::deep {
	class pimpl {

	private:


	public:

		//std::shared_ptr<hv::v1::raii> raii_destructor;

		TF_Buffer* run_options;
		TF_SessionOptions* session_options;
		TF_Graph* graph;
		TF_Status* status;
		TF_Session* session;


		pimpl() {

		}

		~pimpl() {

		}
	};
}




static void deallocator(void* data, size_t length, void* arg) {

}


hv::v1::deep::segmentation::segmentation() : _pimpl(new pimpl()) {
	//run_options(nullptr),
	//	session_options(nullptr),
	//	graph(nullptr),
	//	status(nullptr),
	//	session(nullptr),
	//	is_loaded(false)

	this->_pimpl->run_options = nullptr;
	this->_pimpl->session_options = nullptr;
	this->_pimpl->graph = nullptr;
	this->_pimpl->status = nullptr;
	this->_pimpl->session = nullptr;


	this->_pimpl->run_options = TF_NewBufferFromString("", 0);
	this->_pimpl->session_options = TF_NewSessionOptions();
	this->_pimpl->graph = TF_NewGraph();
	this->_pimpl->status = TF_NewStatus();
	this->_pimpl->session = nullptr;

	//hv::v1::raii* raii = new hv::v1::raii([&] {


	//	});

	//this->_pimpl->raii_destructor = std::shared_ptr<hv::v1::raii>(raii);

}


hv::v1::deep::segmentation::~segmentation() {
	if (this->_pimpl->run_options != nullptr)
		TF_DeleteBuffer(this->_pimpl->run_options);


	if (this->_pimpl->session_options != nullptr)
		TF_DeleteSessionOptions(this->_pimpl->session_options);


	if (this->_pimpl->graph != nullptr)
		TF_DeleteGraph(this->_pimpl->graph);


	if (this->_pimpl->status != nullptr)
		TF_DeleteStatus(this->_pimpl->status);


	if (this->_pimpl->session != nullptr && this->_pimpl->status != nullptr)
		TF_DeleteSession(this->_pimpl->session, this->_pimpl->status);

}

void hv::v1::deep::segmentation::import(std::string path) {

	if (this->is_loaded == true) return;

	model_path = path;

	std::array<char const*, 1> tags = { "serve" };

	this->_pimpl->session = TF_LoadSessionFromSavedModel(this->_pimpl->session_options,
		this->_pimpl->run_options,
		path.c_str(), tags.data(), tags.size(),
		this->_pimpl->graph, nullptr, this->_pimpl->status);

	if (TF_GetCode(this->_pimpl->status) != TF_OK) {
		std::cout << TF_Message(this->_pimpl->status) << '\n';
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

float hv::v1::deep::segmentation::train(float* source_buffer, int source_width, int source_height, int source_channel,
										float* label_buffer, int label_width, int label_height, int label_channel,
										int batch_size) {


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




	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "train_x_input"), 0 });
	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "train_y_label"), 0 });
	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "saver_filename"), 0 });

	output_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "StatefulPartitionedCall_2"), 0 });


	
	//save_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "StatefulPartitionedCall_3"), 0 });

	unsigned int source_size = source_width * source_height * source_channel * batch_size;
	int64_t source_dims[] = { batch_size, source_height, source_width, source_channel };



	unsigned int label_size = label_width * label_height * label_channel * batch_size;
	int64_t label_dims[] = { batch_size, label_height, label_width, label_channel };



	std::string fullPath = model_path;
	fullPath += "variables//variables";

	size_t encoded_size = TF_StringEncodedSize(fullPath.size());
	size_t total_size = 8 + encoded_size;
	std::string encoded_string;
	encoded_string.resize(total_size);

	TF_StringEncode((const char*)fullPath.c_str(), fullPath.size(), (char*)encoded_string.c_str() + 8, encoded_size, this->_pimpl->status);
	if (TF_GetCode(this->_pimpl->status) != TF_OK) {
		std::cerr << "Failed to encode image\n";
		std::cerr << TF_Message(this->_pimpl->status) << std::endl;
		return false;
	}

	int64_t string_dims[] = {1 };


	input_tensors.push_back(TF_NewTensor(TF_FLOAT, source_dims, 4, source_buffer, source_size * sizeof(float), deallocator, 0));
	input_tensors.push_back(TF_NewTensor(TF_FLOAT, label_dims, 4, label_buffer, label_size * sizeof(float), deallocator, 0));
	input_tensors.push_back(TF_NewTensor(TF_STRING, string_dims, 1, (char*)encoded_string.c_str(), encoded_string.size(), deallocator, 0));


	try {
		TF_SessionRun(this->_pimpl->session,
			this->_pimpl->run_options,
			input_vec.data(), input_tensors.data(), input_tensors.size(),
			output_vec.data(), output_tensors.data(), output_vec.size(),
			nullptr, 0,
			nullptr,
			this->_pimpl->status);


		if (TF_GetCode(this->_pimpl->status) != TF_OK) {
			throw std::exception("invalid status");
		}


		const auto data = static_cast<float*>(TF_TensorData(output_tensors[0]));

		return data[0];
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		throw std::exception("run model failed.");
	}




	throw std::exception("run model failed.");

}

void hv::v1::deep::segmentation::run(float* input_buffer, float* output_buffer, int width, int height, int channel, int label) {


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


	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "prediction_x"), 0 });
	output_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "StatefulPartitionedCall_1"), 0 });


	unsigned int input_size = width * height * channel;
	//std::vector<float> input_data;
	//input_data.resize(input_size);
	//float* input_data_ptr = input_data.data();

	//for (int index = 0; index < input_size; index++) {
	//	input_data_ptr[index] = input_buffer[index];
	//}



	int64_t in_dims[] = { 1, height, width, channel };



	//auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII

	input_tensors.push_back(TF_NewTensor(TF_FLOAT, in_dims, 4, input_buffer, input_size * sizeof(float), deallocator, 0));


	try {
		TF_SessionRun(this->_pimpl->session,
			this->_pimpl->run_options,
			input_vec.data(), input_tensors.data(), input_tensors.size(),
			output_vec.data(), output_tensors.data(), output_vec.size(),
			nullptr, 0,
			nullptr,
			this->_pimpl->status);


		if (TF_GetCode(this->_pimpl->status) != TF_OK) {
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




float hv::v1::deep::segmentation::accuracy(float* source_buffer, int source_width, int source_height, int source_channel,
	float* label_buffer, int label_width, int label_height, int label_channel,
	int batch_size) {

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


	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "train_x_input"), 0 });
	input_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "train_y_label"), 0 });
	output_vec.push_back({ TF_GraphOperationByName(this->_pimpl->graph, "StatefulPartitionedCall_2"), 0 });


	unsigned int source_size = source_width * source_height * source_channel * batch_size;
	//std::vector<float> source_data;
	//source_data.resize(source_size);
	//float* source_data_ptr = source_data.data();
	//for (int index = 0; index < source_size; index++) {
	//	source_data_ptr[index] = source_buffer[index];
	//}
	int64_t source_dims[] = { batch_size, source_height, source_width, source_channel };



	unsigned int label_size = label_width * label_height * label_channel * batch_size;
	//std::vector<float> label_data;
	//label_data.resize(label_size);
	//float* label_data_ptr = label_data.data();
	//for (int index = 0; index < label_size; index++) {
	//	label_data_ptr[index] = label_buffer[index];
	//}
	int64_t label_dims[] = { batch_size, label_height, label_width, label_channel };



	//auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII

	input_tensors.push_back(TF_NewTensor(TF_FLOAT, source_dims, 4, source_buffer, source_size * sizeof(float), deallocator, 0));
	input_tensors.push_back(TF_NewTensor(TF_FLOAT, label_dims, 4, label_buffer, label_size * sizeof(float), deallocator, 0));

	try {
		TF_SessionRun(this->_pimpl->session,
			this->_pimpl->run_options,
			input_vec.data(), input_tensors.data(), input_tensors.size(),
			output_vec.data(), output_tensors.data(), output_vec.size(),
			nullptr, 0,
			nullptr,
			this->_pimpl->status);


		if (TF_GetCode(this->_pimpl->status) != TF_OK) {
			throw std::exception("invalid status");
		}

		const auto data = static_cast<float*>(TF_TensorData(output_tensors[0]));

		return data[0];
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		throw std::exception("run model failed.");
	}




	throw std::exception("run model failed.");
}