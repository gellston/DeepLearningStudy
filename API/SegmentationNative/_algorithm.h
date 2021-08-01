#pragma once

/*

#ifndef HV_ALGORITHM
#define HV_ALGORITHM

#include <memory>
#include <string>


#if HVVAPI
#define HVAPI_EXPORT __declspec(dllexport)
//#define HVAPI_TEMPLATE_EXPORT
#else
#define HVAPI_EXPORT __declspec(dllimport)
//#define HVAPI_TEMPLATE_EXPORT extern
#endif


typedef struct _Point32
{
	double x;
	double y;
}PT32;


typedef struct _Line {
	double mx, my;
	double cx, cy;
	double dTheta;
}LINE;


typedef struct _MeasurementLine
{
	PT32	center;
	double	angle;
	double	range;
	double	distance;
	bool	direction;
	int		type;
	double  threshold;
}MEASUREMENT_LINE;





namespace hv::v1::algorithm {
	class HVAPI_EXPORT algorithm {
	private:

	public:
		algorithm();
		~algorithm();

		bool findLine(unsigned char* buffer, int image_width, int image_height, MEASUREMENT_LINE _inLineParam, LINE* _outLine);
		bool getCrossPointFromTwoLine(LINE _inLine1, LINE _inLine2, PT32* _outCrossPoint);
		double getAlignmentDegreeFromDegree(double _inSrcDegree, double _inBaseDegree);
	};
};




#endif // !HV_DEEP
*/