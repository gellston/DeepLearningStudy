#pragma once

#ifndef HV_DEEP
#define HV_DEEP

#include "raii.h"



#include <memory>
#include <string>
#include <tensorflow/c/c_api.h>

namespace hv::v1::deep{
	
	class segmentation {
	private:
		std::shared_ptr<hv::v1::raii> raii_destructor;

		TF_Buffer* run_options;
		TF_SessionOptions* session_options;
		TF_Graph* graph;
		TF_Status* status;
		TF_Session* session;


		//TF_Operation* input_op;
		//TF_Operation* output_op;

		bool is_loaded;

	public:
		segmentation();
		~segmentation();

		void import(std::string path);
		void run(unsigned char* input_buffer, unsigned char* output_buffer, int width, int height, int channel, int label);

		const char* tf_version();
	};

}

#endif // !HV_DEEP