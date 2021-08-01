#pragma once

#pragma unmanaged
#include <deep.h>





#pragma managed
#include <msclr/marshal_cppstd.h>

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

				Segmentation();

				~Segmentation();
				!Segmentation();

				void Import(System::String^ path);
				void Run(IntPtr input_buffer, IntPtr output_buffer, int width, int height, int channel, int label);


				
			};
		}

	}
}
