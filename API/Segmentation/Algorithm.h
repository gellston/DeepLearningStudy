#pragma once


#include <msclr/marshal_cppstd.h>


#include <_algorithm.h>




#include "managed_shared_ptr.h"



using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;


using namespace System;

namespace HV {
	namespace V1 {
		namespace ALGORITHM {

			public ref class MeasurementLine {
			private:
				double _centerX;
				double _centerY;
				double _angle;
				double _range;
				double _distance;
				bool _direction;
				int _type;
				double _threshold;

			public:
				MeasurementLine() {
					_centerX = 0;
					_centerY = 0;
					_angle = 0;
					_range = 0;
					_distance = 0;
					_direction = 0;
					_type = 0;
					_threshold = 0;
				}
				~MeasurementLine() { this->!MeasurementLine(); }
				!MeasurementLine() {  }


				property double CenterX {
					double get() {
						return this->_centerX;
					}

					void set(double value) {
						this->_centerX = value;
					}
				}

				property double CenterY {
					double get() {
						return this->_centerY;
					}

					void set(double value) {
						this->_centerY = value;
					}
				}

				property double Angle {
					double get() {
						return this->_angle;
					}

					void set(double value) {
						this->_angle = value;
					}
				}

				property double Range {
					double get() {
						return this->_range;
					}

					void set(double value) {
						this->_range = value;
					}
				}

				property double Distance {
					double get() {
						return this->_distance;
					}

					void set(double value) {
						this->_distance = value;
					}
				}


				property bool Direction {
					bool get() {
						return this->_direction;
					}

					void set(bool value) {
						this->_direction = value;
					}
				}

				property int Type {
					int get() {
						return this->_type;
					}

					void set(int value) {
						this->_type = value;
					}
				}

				property double Threadhold {
					double get() {
						return this->_threshold;
					}

					void set(double value) {
						this->_threshold = value;
					}
				}


			};


			public ref class Line {
			private:
				double _mx;
				double _my;
				double _cx;
				double _cy;
				double _dTheta;

			public:
				Line() { 
					_mx = 0;
					_my = 0;
					_cx = 0;
					_cy = 0;
					_dTheta = 0;

				}
				~Line() { this->!Line(); }
				!Line(){  }


				property double MX {
					double get() {
						return this->_mx;
					}

					void set(double value) {
						this->_mx = value;
					}
				}

				property double MY {
					double get() {
						return this->_my;
					}

					void set(double value) {
						this->_my = value;
					}
				}

				property double CX {
					double get() {
						return this->_cx;
					}

					void set(double value) {
						this->_cx = value;
					}
				}

				property double CY {
					double get() {
						return this->_cy;
					}

					void set(double value) {
						this->_cy = value;
					}
				}

				property double Theta {
					double get() {
						return this->_dTheta;
					}

					void set(double value) {
						this->_dTheta = value;
					}
				}

			
			};

			public ref class Algorithm
			{
			internal:

				HV::V1::mananged_shared_ptr<hv::v1::algorithm::algorithm> _instance;

			public:

				Algorithm() : _instance(new hv::v1::algorithm::algorithm()) {

				}

				~Algorithm() {
					this->!Algorithm();
				}
				!Algorithm() {
					this->_instance.~mananged_shared_ptr();
				}

				bool GetCrossPointFromTwoLine(Line^ _inLine1, Line^ _inLine2, [Out]double% pointX, [Out]double% pointY) {
					LINE _line_1;
					LINE _line_2;

					_line_1.cx = _inLine1->CX;
					_line_1.cy = _inLine1->CY;
					_line_1.dTheta = _inLine1->Theta;
					_line_1.mx = _inLine1->MX;
					_line_1.my = _inLine1->MY;

					_line_2.cx = _inLine2->CX;
					_line_2.cy = _inLine2->CY;
					_line_2.dTheta = _inLine2->Theta;
					_line_2.mx = _inLine2->MX;
					_line_2.my = _inLine2->MY;

					PT32 point;

					auto returnValue = this->_instance->getCrossPointFromTwoLine(_line_1, _line_2, &point);

					
					pointX = point.x;
					pointY = point.y;

					return returnValue;

				}

				double GetAlignmentDegreeFromDegree(double inAngle, double baseAngle) {
					return this->_instance->getAlignmentDegreeFromDegree(inAngle, baseAngle);
				}

				//[Out]int% value
				bool FindLine(IntPtr buffer, int image_width, int image_height, MeasurementLine^ inLineParam, [Out]Line^% outLine) {

					auto input_pointer = static_cast<unsigned char*>(buffer.ToPointer());
					
					LINE outline;


					MEASUREMENT_LINE inLine;
					inLine.center.x = inLineParam->CenterX;
					inLine.center.y = inLineParam->CenterY;
					inLine.angle = inLineParam->Angle;
					inLine.distance = inLineParam->Distance;
					inLine.direction = inLineParam->Direction;
					inLine.range = inLineParam->Range;
					inLine.threshold = inLineParam->Threadhold;
					inLine.type = inLineParam->Type;


					bool returnValue = this->_instance->findLine(input_pointer, image_width, image_height, inLine, &outline);


					Line^ outLineSharp = gcnew Line();
					outLineSharp->CX = outline.cx;
					outLineSharp->CY = outline.cy;
					outLineSharp->MX = outline.mx;
					outLineSharp->MY = outline.my;
					outLineSharp->Theta = outline.dTheta;


					outLine = outLineSharp;

					return returnValue;
				}

			};
		}

	}
}
