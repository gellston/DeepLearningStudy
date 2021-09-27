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




namespace deep {
	class pimpl;
}

namespace deep {

	class HVAPI_EXPORT classification {
	private:
		std::shared_ptr<pimpl> _pimpl;

		bool is_loaded;

	public:
		classification();
		~classification();

		void import(std::string path);
		void run(unsigned char* input_buffer, float* output_buffer, int width, int height, int channel, int label);

	};



}



#endif // !HV_DEEP