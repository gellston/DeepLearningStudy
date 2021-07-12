#pragma once


#include <msclr/marshal_cppstd.h>


#include <deep.h>




#include "managed_shared_ptr.h"



using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;


using namespace System;

namespace HV {
	namespace V1 {
		namespace DEEP {
			public ref class Segmentation
			{
			internal:

				HV::V1::mananged_shared_ptr<hv::v1::deep::segmentation> _instance;

			public:

				Segmentation() : _instance(new hv::v1::deep::segmentation()) {

				}

				~Segmentation() {
					this->!Segmentation();
				}
				!Segmentation() {
					this->_instance.~mananged_shared_ptr();
				}

				void Import(System::String^ path) {
					auto stdPath = msclr::interop::marshal_as<std::string>(path);
					try {
						this->_instance->import(stdPath);
					}
					catch (std::exception e) {
						throw gcnew System::Exception(gcnew System::String(e.what()));
					}
					

				}

				void Run(IntPtr input_buffer, IntPtr output_buffer, int width, int height, int channel, int label) {

					auto input_pointer = static_cast<unsigned char*>(input_buffer.ToPointer());
					auto output_pointer = static_cast<unsigned char*>(output_buffer.ToPointer());


					try {
						this->_instance->run(input_pointer, output_pointer, width, height, channel, label);
					}
					catch (std::exception e) {
						throw gcnew System::Exception(gcnew System::String(e.what()));
					}
					
				}


				
			};
		}

	}
}
