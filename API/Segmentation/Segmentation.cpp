
#include "Segmentation.h"



using namespace System;

HV::V1::DEEP::Segmentation::Segmentation() :_instance(new hv::v1::deep::segmentation()) {

}

HV::V1::DEEP::Segmentation::~Segmentation() {
	this->!Segmentation();
}

HV::V1::DEEP::Segmentation::!Segmentation() {
	this->_instance.~mananged_shared_ptr();
}


void HV::V1::DEEP::Segmentation::Import(System::String^ path) {
	auto stdPath = msclr::interop::marshal_as<std::string>(path);
	try {
		this->_instance->import(stdPath);
	}
	catch (std::exception e) {
		throw gcnew System::Exception(gcnew System::String(e.what()));
	}


}

void HV::V1::DEEP::Segmentation::Run(IntPtr input_buffer, IntPtr output_buffer, int width, int height, int channel, int label) {

	auto input_pointer = static_cast<unsigned char*>(input_buffer.ToPointer());
	auto output_pointer = static_cast<unsigned char*>(output_buffer.ToPointer());


	try {
		this->_instance->run(input_pointer, output_pointer, width, height, channel, label);
	}
	catch (std::exception e) {
		throw gcnew System::Exception(gcnew System::String(e.what()));
	}

}