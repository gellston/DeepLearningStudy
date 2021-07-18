#pragma once

#ifndef HV_DEEP
#define HV_DEEP

#include <memory>
#include <string>


#if HVVAPI
#define HVAPI_EXPORT __declspec(dllexport)
//#define HVAPI_TEMPLATE_EXPORT
#else
#define HVAPI_EXPORT __declspec(dllimport)
//#define HVAPI_TEMPLATE_EXPORT extern
#endif




namespace hv::v1::deep {
	class pimpl;
}

namespace hv::v1::deep {

	class HVAPI_EXPORT segmentation {
	private:
		std::shared_ptr<pimpl> _pimpl;
		std::string model_path;
		bool is_loaded;

	public:
		segmentation();
		~segmentation();

		void import(std::string path);
		void run(float* input_buffer, float* output_buffer, int width, int height, int channel, int label);
		float train(float* input_buffer, int input_width, int input_height, int input_channel,
			float* output_buffer, int output_width, int output_height, int output_channel,
			int batch_size);
		float accuracy(float* input_buffer, int input_width, int input_height, int input_channel,
			float* output_buffer, int output_width, int output_height, int output_channel,
			int batch_size);
	};



}



#endif // !HV_DEEP