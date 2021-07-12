

#include "_algorithm.h"
#include <vector>
#include <math.h>

#define PI 3.141592653589793


#define WIDTHBYTES(bits) ((DWORD)(((bits)+31) & (~31)) / 8)
#define SAFE_DELETE(p) if(p){ delete p; p=NULL; }

#define ROUND(A)	((A)>=0 ? int(0.5+(A)) : int((A)-0.5))
#define PI 3.141592653589793

#define CHK_MIN(p, minV ) if( p < minV ) p = minV;
#define CHK_MAX(p, maxV ) if( p > maxV ) p = maxV;
#define CHK_RANGE(minV, p, maxV ) if( p < minV ) { p=minV; } else if( p > maxV ) { p = maxV; }

#define CLIP(val, low, high) {if(val<low) val=low; if(val>high) val=high;}

#define MAKE_RET_COLOR(r,g,b)					(r*1000000)+(g*1000)+(b)

#define WAY_MAX			720
#define STATUS_NONE		-1
#define STATUS_BLACK	0
#define STATUS_WHITE	1
#define STATUS_ANYTHING	2

//#ifndef max
//#define max(a,b)            (((a) > (b)) ? (a) : (b))
//#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif





int GetPointApplyAngleDistance(PT32 _inCenter, double _inRadius, double _inAngle, PT32 * _outPt);
double GetAlignmentDegreeFromDegree(double _inSrcDegree, double _inBaseDegree);
int GetDistanceLineToPoint(LINE _Line, PT32 _inPt1, double* _outDistance, PT32 * _outNormalCrossPoint);
int GetDistanceLineToPoint(PT32 _inLinePt1, PT32 _inLinePt2, PT32 _inPt, double* _outDistance, PT32 * _outNormalCrossPoint);
int GetDistanceBetweenPoints(PT32 _inPt1, PT32 _inPt2, double* _outDistance);
int GetAngleFromPoints(PT32 _inCenter, PT32 _inPoint, double* _outAngle);
void _RANSAC_GetSamples(PT32 * _inSamples, int _inSampleTimes, PT32 * _outData, int _inNoData);
bool _RANSAC_FindSamples(PT32 * _inSamples, int no_samples, PT32 * _outData);
double _RANSAC_ComputeDistanceLine(LINE & line, PT32 & pt);
int _RANSAC_ComputeModelParamLine(PT32 _inSamples[], int no_samples, LINE & _outModel);
double _RANSAC_ModelVerificationLine(PT32 * inliers, int* no_inliers, LINE & estimated_model, PT32 * data, int no_data, double distance_threshold);
double _RANSAC_LINE(int width, int height, PT32 * _inSamples, int no_data, LINE & model, double distance_threshold);
int GetCrossPointFromTwoLine(LINE _inLine1, LINE _inLine2, PT32 * _outCrossPoint);




int GetCrossPointFromTwoLine(LINE _inLine1, LINE _inLine2, PT32 * _outCrossPoint)
{
	double s, t, _s, _t, under;
	PT32 p1, p2, p3, p4;

	s = t = _s = _t = under = 0.0;


	_inLine1.mx = cos(_inLine1.dTheta / (180 / PI));
	_inLine1.my = sin(_inLine1.dTheta / (180 / PI));
	
	_inLine2.mx = cos(_inLine2.dTheta / (180 / PI));
	_inLine2.my = sin(_inLine2.dTheta / (180 / PI));


	p1.x = _inLine1.cx;
	p1.y = _inLine1.cy;
	p2.x = _inLine1.cx + 10 * _inLine1.mx;
	p2.y = _inLine1.cy + 10 * _inLine1.my;
	p3.x = _inLine2.cx;
	p3.y = _inLine2.cy;
	p4.x = _inLine2.cx + 10 * _inLine2.mx;
	p4.y = _inLine2.cy + 10 * _inLine2.my;

	under = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
	if (under == 0) return false;

	_t = (p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x);
	_s = (p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x);

	t = _t / under;
	s = _s / under;

	_outCrossPoint->x = p1.x + t * (double)(p2.x - p1.x);
	_outCrossPoint->y = p1.y + t * (double)(p2.y - p1.y);

	return true;
}




int GetPointApplyAngleDistance(PT32 _inCenter,						// _inCetner 기준으로 부터
	double _inRadius,						// _inRadius 만큼 거리에
	double _inAngle,						// _inAngle  의 각도로 옮겨져 있는 
	PT32* _outPt)							// 좌표 결과
{
	_outPt->x = _inCenter.x + _inRadius * cos(_inAngle * PI / 180);	// RADIAN
	_outPt->y = _inCenter.y + _inRadius * sin(_inAngle * PI / 180);

	return 1;
}


double GetAlignmentDegreeFromDegree(double _inSrcDegree, double _inBaseDegree)
{
	double src1 = _inSrcDegree - 180.0;
	double src2 = _inSrcDegree;
	double src3 = _inSrcDegree + 180.0;
	double src4 = _inSrcDegree - 360.0;
	double src5 = _inSrcDegree + 360.0;

	double diff1 = fabs(src1 - _inBaseDegree);
	double diff2 = fabs(src2 - _inBaseDegree);
	double diff3 = fabs(src3 - _inBaseDegree);
	double diff4 = fabs(src4 - _inBaseDegree);
	double diff5 = fabs(src5 - _inBaseDegree);
	double diffMin = min(min(min(diff1, diff4), min(diff2, diff3)), diff5);

	if (diffMin == diff1)
		return src1;
	else if (diffMin == diff2)
		return src2;
	else if (diffMin == diff3)
		return src3;
	else if (diffMin == diff4)
		return src4;
	else
		return src5;
}


int GetDistanceLineToPoint(LINE _Line, PT32 _inPt1, double* _outDistance, PT32* _outNormalCrossPoint)
{
	PT32 l1, l2;
	PT32 lc;
	lc.x = _Line.cx;
	lc.y = _Line.cy;
	double lineLength = 100000;
	GetPointApplyAngleDistance(lc, lineLength, _Line.dTheta, &l1);
	GetPointApplyAngleDistance(lc, lineLength, _Line.dTheta + 180.0, &l2);

	GetDistanceLineToPoint(l1, l2, _inPt1, _outDistance, _outNormalCrossPoint);

	return 1;
}




int GetDistanceLineToPoint(PT32 _inLinePt1, PT32 _inLinePt2, PT32 _inPt, double* _outDistance, PT32* _outNormalCrossPoint)
{
	double X1 = _inLinePt1.x;
	double Y1 = _inLinePt1.y;
	double X2 = _inLinePt2.x;
	double Y2 = _inLinePt2.y;

	double dx = X2 - X1;
	double dy = Y2 - Y1;

	double ptX = _inPt.x;
	double ptY = _inPt.y;
	double nearX, nearY;

	if (dx == 0 && dy == 0)
	{
		dx = ptX - X1;
		dy = ptY - Y1;

		nearX = X1;
		nearY = Y1;

		*_outDistance = sqrt((double)(dx * dx + dy * dy));
		_outNormalCrossPoint->x = nearX;
		_outNormalCrossPoint->y = nearY;

		return 1;
	}

	double t = (double)((ptX - X1) * dx + (ptY - Y1) * dy) / (double)(dx * dx + dy * dy);

	if (t < 0)
	{
		dx = ptX - X1;
		dy = ptY - Y1;
		nearX = X1;
		nearY = Y1;
	}
	else if (t > 1)
	{
		dx = ptX - X2;
		dy = ptY - Y2;
		nearX = X2;
		nearY = Y2;
	}
	else
	{
		nearX = X1 + t * dx;
		nearY = Y1 + t * dy;
		dx = ptX - nearX;
		dy = ptY - nearY;
	}

	*_outDistance = sqrt((double)(dx * dx + dy * dy));
	_outNormalCrossPoint->x = nearX;
	_outNormalCrossPoint->y = nearY;

	return 1;
}


// raw code
int GetDistanceBetweenPoints(PT32 _inPt1,						//			  _outDistance
	PT32 _inPt2,						//		Pt1 --------------- Pt2
	double* _outDistance)			//
{
	*_outDistance = sqrt((double)(pow(_inPt2.x - _inPt1.x, 2) + pow(_inPt2.y - _inPt1.y, 2)));

	return 1;
}
//	     |      /_inPoint ( return -45 )
int GetAngleFromPoints(PT32 _inCenter,					//       |     /					
	PT32 _inPoint,						//		 |    /		
	double* _outAngle)				//	 _inCenter -------------- 0			
{
	*_outAngle = atan2(_inPoint.y - _inCenter.y, _inPoint.x - _inCenter.x) * 180 / PI;
	if (*_outAngle < 0) *_outAngle += 360.0;
	return 1;
}


// Recipe Control
//int GetPointApplyAngleDistance(int _innX, int _innY, int _innRadius, int _innAngle, int _nOutPt)
//{
//	PT32 _inCenter, _outPT;
//	_inCenter.x = RetVariableGetValue(_innX);
//	_inCenter.y = RetVariableGetValue(_innY);
//	double _inRad = RetVariableGetValue(_innRadius);
//	double _inAngle = RetVariableGetValue(_innAngle);
//
//	int _Ret = GetPointApplyAngleDistance(_inCenter, _inRad, _inAngle, &_outPT);
//
//	if (_Ret == RET_OK)
//	{
//		RetVariableSet(_nOutPt, "Rotated Point X", _outPT.x);
//		RetVariableSet(_nOutPt + 1, "Rotated Point Y", _outPT.y);
//	}
//
//	return RET_OK;
//}



void _RANSAC_GetSamples(PT32* _inSamples, int _inSampleTimes, PT32* _outData, int _inNoData)
{
	// 데이터에서 중복되지 않게 _inSampleTimes개의 무작위 셈플을 채취한다.
	for (int i = 0; i < _inSampleTimes; )
	{
		int j = rand() % _inNoData;

		if (!_RANSAC_FindSamples(_inSamples, i, &_outData[j]))
		{
			_inSamples[i] = _outData[j];
			++i;
		}
	}
}

bool _RANSAC_FindSamples(PT32* _inSamples, int no_samples, PT32* _outData)
{
	for (int i = 0; i < no_samples; ++i)
	{
		if (_inSamples[i].x == _outData->x && _inSamples[i].y == _outData->y)
		{
			//return RET_OK;
			return true;
		}
	}

	//return RET_FAIL;
	return false;
}

double _RANSAC_ComputeDistanceLine(LINE& line, PT32& pt)
{
	// 한 점(pt)로부터 직선(line)에 내린 수선의 길이(distance)를 계산한다.
	return fabs((pt.x - line.cx) * line.my - (pt.y - line.cy) * line.mx) / sqrt(line.mx * line.mx + line.my * line.my);
}



int _RANSAC_ComputeModelParamLine(PT32 _inSamples[], int no_samples, LINE& _outModel)
{
	// PCA 방식으로 직선 모델의 파라메터를 예측한다.

	double sx = 0, sy = 0;
	double sxx = 0, syy = 0;
	double sxy = 0, sw = 0;

	for (int i = 0; i < no_samples; ++i)
	{
		double& x = _inSamples[i].x;
		double& y = _inSamples[i].y;

		sx += x;
		sy += y;
		sxx += x * x;
		sxy += x * y;
		syy += y * y;
		sw += 1;
	}

	//variance;
	double vxx = (sxx - sx * sx / sw) / sw;
	double vxy = (sxy - sx * sy / sw) / sw;
	double vyy = (syy - sy * sy / sw) / sw;

	//principal axis
	double theta = atan2(2 * vxy, vxx - vyy) / 2;

	_outModel.mx = cos(theta);
	_outModel.my = sin(theta);
	_outModel.dTheta = theta * 180 / PI;

	//center of mass(xc, yc)
	_outModel.cx = sx / sw;
	_outModel.cy = sy / sw;

	//직선의 방정식: sin(theta)*(x - sx) = cos(theta)*(y - sy);
	return 1;
}

double _RANSAC_ModelVerificationLine(PT32* inliers, int* no_inliers, LINE& estimated_model, PT32* data, int no_data, double distance_threshold)
{
	*no_inliers = 0;

	double cost = 0.;

	for (int i = 0; i < no_data; i++) {
		// 직선에 내린 수선의 길이를 계산한다.
		double distance = _RANSAC_ComputeDistanceLine(estimated_model, data[i]);

		// 예측된 모델에서 유효한 데이터인 경우, 유효한 데이터 집합에 더한다.
		if (distance < distance_threshold) {
			cost += 1.;

			inliers[*no_inliers] = data[i];
			++(*no_inliers);
		}
	}

	return cost;
}



double _RANSAC_LINE(int width, int height, PT32* _inSamples, int no_data, LINE& model, double distance_threshold)
{
	const int no_samples = 2;

	if (no_data < no_samples)
	{
		return 0.;
	}

	double max_cost = 0.;

	int max_iteration = (int)(1 + log(1. - 0.99) / log(1. - pow(0.5, no_samples))) * 100;
	max_iteration = std::max(1, max_iteration);
	

	// 	double *costs = new double[max_iteration];
	// 	memset(costs, 0, sizeof(double)*max_iteration);
	// 	LINE *models = new LINE[max_iteration];
	// 	memset(models, 0, sizeof(LINE)*max_iteration);


	//	Concurrency::parallel_for(0, (int)max_iteration, [&](int i)

	for (int i = 0; i < max_iteration; i++)
	{
		// 1. hypothesis
		PT32* samples = new PT32[no_samples];
		int no_inliers = 0;
		PT32* inliers = new PT32[no_data];

		LINE estimated_model;

		// 원본 데이터에서 임의로 N개의 셈플 데이터를 고른다.
		_RANSAC_GetSamples(samples, no_samples, _inSamples, no_data);

		// 이 데이터를 정상적인 데이터로 보고 모델 파라메터를 예측한다.
		_RANSAC_ComputeModelParamLine(samples, no_samples, estimated_model);

		// 2. Verification

		// 원본 데이터가 예측된 모델에 잘 맞는지 검사한다.
		double cost = _RANSAC_ModelVerificationLine(inliers, &no_inliers, estimated_model, _inSamples, no_data, distance_threshold);

		//// 만일 예측된 모델이 잘 맞는다면, 이 모델에 대한 유효한 데이터로 새로운 모델을 구한다.
		if (max_cost < cost)
		{
			max_cost = cost;
			_RANSAC_ComputeModelParamLine(inliers, no_inliers, model);
		}

		delete[] samples;
		delete[] inliers;
	}


	// 3. Final Model Search
// 	for (int i = 0; i < max_iteration; i++)
// 	{
// 		if (max_cost < costs[i])
// 		{
// 			max_cost = costs[i];
// 			model = models[i];
// 		}
// 	}
// 
// 	delete[] costs;
// 	delete[] models;


	// 4. align x, y data of key points
	double leftDistance = 0;
	double rightDistance = 0;
	double KeyDistance = 0;
	PT32 leftPt, rightPt;
	leftPt.x = _inSamples[0].x;
	leftPt.y = _inSamples[0].y;

	rightPt.x = _inSamples[1].x;
	rightPt.y = _inSamples[1].y;

	PT32 cpt, keyPt;
	cpt.x = model.cx;
	cpt.y = model.cy;

	double mina, maxa;
	mina = model.dTheta - 90;
	if (mina < 0.0) mina += 360;
	maxa = model.dTheta + 90;
	if (maxa > 360.0) maxa -= 360;

	for (int i = 0; i < no_data; i++) {
		// 직선에 내린 수선의 길이를 계산한다.
		double distance = _RANSAC_ComputeDistanceLine(model, _inSamples[i]);

		// 허용 threshold 이내 임을 확인
		if (distance < distance_threshold)
		{
			// left & right 기준으로 angle을 나눔
			double angle = 0.0;
			GetAngleFromPoints(cpt, _inSamples[i], &angle);

			// 해당 point와 현재 model 중심과의 거리 축정
			GetDistanceBetweenPoints(cpt, _inSamples[i], &distance);

			bool bRight = false;
			if (mina > maxa)
			{
				if (angle < maxa || mina < angle)
					bRight = true;
			}
			else
			{
				if (mina < angle && angle < maxa)
					bRight = true;
			}

			//if (mina < angle || angle < maxa)
			if (bRight)
			{
				if (distance > rightDistance) {
					rightDistance = distance;
					rightPt = _inSamples[i];
				}
			}
			else
			{
				if (distance > leftDistance) {
					leftDistance = distance;
					leftPt = _inSamples[i];
				}
			}
		}
	}

	// 좌우 가장 먼 point 들으 중간 위치 획득
	keyPt.x = (leftPt.x + rightPt.x) / 2.0;
	keyPt.y = (leftPt.y + rightPt.y) / 2.0;

	// 라인위의 임의의 좌우 끝 점 계산 
	GetPointApplyAngleDistance(cpt, width, model.dTheta, &leftPt);
	GetPointApplyAngleDistance(cpt, width, model.dTheta + 180, &rightPt);

	// update Center Position
	GetDistanceLineToPoint(leftPt, rightPt, keyPt, &KeyDistance, &cpt);
	model.cx = cpt.x;
	model.cy = cpt.y;


	return max_cost;
}




hv::v1::algorithm::algorithm::algorithm() {

}

hv::v1::algorithm::algorithm::~algorithm() {

}

bool hv::v1::algorithm::algorithm::getCrossPointFromTwoLine(LINE _inLine1, LINE _inLine2, PT32* _outCrossPoint) {


	return GetCrossPointFromTwoLine(_inLine1, _inLine2, _outCrossPoint);
}

bool hv::v1::algorithm::algorithm::findLine(unsigned char* buffer, int image_width, int image_height, MEASUREMENT_LINE _inLineParam, LINE* _outLine) {
	if (buffer == nullptr || image_width < 1 || image_height < 1)
		return false;

	//_inLineParam.center.x -= m_ptROIOffset.x;
	//_inLineParam.center.y -= m_ptROIOffset.y;

	PT32 _SearchLinePtStart_1, _SearchLinePtStart_2;
	PT32 _SearchLinePtEnd_1, _SearchLinePtEnd_2;
	PT32 _inLineEndPt1, _inLineEndPt2;

	GetPointApplyAngleDistance(_inLineParam.center, _inLineParam.range / 2, _inLineParam.angle + 180, &_inLineEndPt1);
	GetPointApplyAngleDistance(_inLineParam.center, _inLineParam.range / 2, _inLineParam.angle, &_inLineEndPt2);

	GetPointApplyAngleDistance(_inLineEndPt1, _inLineParam.distance / 2, _inLineParam.angle + 180 + 90, &_SearchLinePtStart_1);
	GetPointApplyAngleDistance(_inLineEndPt2, _inLineParam.distance / 2, _inLineParam.angle - 90, &_SearchLinePtStart_2);
	GetPointApplyAngleDistance(_inLineEndPt1, _inLineParam.distance / 2, _inLineParam.angle + 180 - 90, &_SearchLinePtEnd_1);
	GetPointApplyAngleDistance(_inLineEndPt2, _inLineParam.distance / 2, _inLineParam.angle + 90, &_SearchLinePtEnd_2);


	//SetStepRetImage(TYPE_STEP_INPUT, m_ImgWork);
	//SetStepRetGUI();

	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_POINT2, MAKE_RET_COLOR(255, 0, 0), _SearchLinePtStart_1.x, _SearchLinePtStart_1.y, 0, 0, 0, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_POINT2, MAKE_RET_COLOR(255, 0, 0), _SearchLinePtEnd_1.x, _SearchLinePtEnd_1.y, 0, 0, 0, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_POINT2, MAKE_RET_COLOR(0, 0, 255), _SearchLinePtStart_2.x, _SearchLinePtStart_2.y, 0, 0, 0, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_POINT2, MAKE_RET_COLOR(0, 0, 255), _SearchLinePtEnd_2.x, _SearchLinePtEnd_2.y, 0, 0, 0, ""));

	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_LINE1, MAKE_RET_COLOR(0, 0, 255), _SearchLinePtStart_1.x, _SearchLinePtStart_1.y, 0, _SearchLinePtEnd_1.x, _SearchLinePtEnd_1.y, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_LINE1, MAKE_RET_COLOR(0, 0, 255), _SearchLinePtStart_2.x, _SearchLinePtStart_2.y, 0, _SearchLinePtEnd_2.x, _SearchLinePtEnd_2.y, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_LINE1, MAKE_RET_COLOR(0, 255, 0), _SearchLinePtStart_1.x, _SearchLinePtStart_1.y, 0, _SearchLinePtStart_2.x, _SearchLinePtStart_2.y, ""));
	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_LINE1, MAKE_RET_COLOR(0, 255, 0), _SearchLinePtEnd_1.x, _SearchLinePtEnd_1.y, 0, _SearchLinePtEnd_2.x, _SearchLinePtEnd_2.y, ""));

	//AddStepRetMarker(TYPE_STEP_INPUT, MakeMarker(TYPE_MAKER_TRIANGLE, MAKE_RET_COLOR(0, 0, 255), _inLineParam.center.x, _inLineParam.center.y, (_inLineParam.direction == false ? 0 : 1), _inLineParam.angle, _inLineParam.type, ""));

	//SetStepRetImage(TYPE_STEP_OUTPUT, m_ImgWork);

	PT32 SearchStartPT;
	double SearchAngle;

	if (_inLineParam.direction == true)
	{
		SearchStartPT = _SearchLinePtStart_1;
		SearchAngle = _inLineParam.angle + 90;
	}
	else
	{
		SearchStartPT = _SearchLinePtEnd_1;
		SearchAngle = _inLineParam.angle + 270;
	}

	std::vector<PT32> fPoints;
	PT32 searchPT;
	PT32 fpoint;

	for (int i = 0; i < _inLineParam.range; i++)
	{
		// SearchStartPT 를 증가하면서 변경
		GetPointApplyAngleDistance(SearchStartPT, i, _inLineParam.angle, &searchPT);

		// 초기 값 획득
		double preVal, curVal;
		PT32 prePT, curPT;

		prePT = curPT = searchPT;

		int rx = 0;
		int ry = 0;
		double dx = 0;
		double dy = 0;

		int idx1 = (ry)*image_width + rx;
		int idx2 = (ry + 1) * image_width + rx;
		int idx3 = (ry)*image_width + (rx + 1);
		int idx4 = (ry + 1) * image_width + (rx + 1);

		// 영상 영역 내Start 찾기
		bool bFound = false;
		double sr = 0;
		for (sr = 0; sr < _inLineParam.distance; sr += 1)
		{
			GetPointApplyAngleDistance(searchPT, sr, SearchAngle, &curPT);

			if (curPT.x < 1 || curPT.y < 1 || curPT.x >= image_width - 1 || curPT.y >= image_height - 1)
				continue;

			rx = (int)(curPT.x);
			ry = (int)(curPT.y);
			dx = curPT.x - rx;
			dy = curPT.y - ry;

			idx1 = (ry)*image_width + rx;
			idx2 = (ry + 1) * image_width + rx;
			idx3 = (ry)*image_width + (rx + 1);
			idx4 = (ry + 1) * image_width + (rx + 1);

			curVal = (dx * dy) * (double)buffer[idx4] +
				(dx * (1 - dy)) * (double)buffer[idx3] +
				((1 - dx) * dy) * (double)buffer[idx2] +
				((1 - dx) * (1 - dy)) * (double)buffer[idx1];

			preVal = curVal;
			prePT = curPT;

			bFound = true;

			break;
		}

		if (bFound == false)
			continue;

		double Delta = 0, maxDelta = 0;

		for (sr; sr < _inLineParam.distance; sr += 1)
		{
			GetPointApplyAngleDistance(searchPT, sr, SearchAngle, &curPT);

			if (curPT.x < 1 || curPT.y < 1 || curPT.x >= image_width - 1 || curPT.y >=image_height - 1)
				continue;

			rx = (int)(curPT.x);
			ry = (int)(curPT.y);
			dx = curPT.x - rx;
			dy = curPT.y - ry;

			idx1 = (ry)*image_width + rx;
			idx2 = (ry + 1) * image_width + rx;
			idx3 = (ry)*image_width + (rx + 1);
			idx4 = (ry + 1) * image_width + (rx + 1);


			curVal = (dx * dy) * (double)buffer[idx4] +
				(dx * (1 - dy)) * (double)buffer[idx3] +
				((1 - dx) * dy) * (double)buffer[idx2] +
				((1 - dx) * (1 - dy)) * (double)buffer[idx1];

			if (_inLineParam.type == 1)
				Delta = -(curVal - preVal);
			else
				Delta = curVal - preVal;

			if (Delta > maxDelta)
			{
				fpoint = prePT;
				maxDelta = Delta;
			}

			preVal = curVal;
			prePT = curPT;
		}

		if (maxDelta > 1)
		{
			fPoints.push_back(fpoint);
		}
	}


	if ((int)fPoints.size() <= 0)
	{
		std::vector<PT32>().swap(fPoints);
		fPoints.clear();

		return false;
	}


	PT32* inLinePoints = new PT32[(int)fPoints.size()];
	for (int i = 0; i < (int)fPoints.size(); i++)
	{
		inLinePoints[i].x = fPoints[i].x;
		inLinePoints[i].y = fPoints[i].y;

		//AddStepRetMarker(TYPE_STEP_OUTPUT,
		//	MakeMarker(TYPE_MAKER_POINT2, MAKE_RET_COLOR(0, 255, 0), fPoints[i].x, fPoints[i].y, 0, 0, 0, ""));
	}

	CHK_MIN(_inLineParam.threshold, 0.1);

	LINE outLine;
	double ret = _RANSAC_LINE(image_width, image_height, inLinePoints, (int)fPoints.size(), outLine, _inLineParam.threshold);

	SAFE_DELETE(inLinePoints);

	if (ret < 3)
	{
		std::vector<PT32>().swap(fPoints);
		fPoints.clear();

		return false;
	}

	*_outLine = outLine;
	_outLine->dTheta = _outLine->dTheta < 0 ? _outLine->dTheta + 360.0 : _outLine->dTheta;
	_outLine->dTheta = GetAlignmentDegreeFromDegree(_outLine->dTheta, _inLineParam.angle);
	while (_outLine->dTheta > 360.0)
		_outLine->dTheta -= 360.0;

	//AddStepRetMarker(TYPE_STEP_OUTPUT, MakeMarker(TYPE_MAKER_POINT1, MAKE_RET_COLOR(255, 0, 0), outLine.cx, outLine.cy, 0, 0, 0, ""));

	PT32 drawline1, drawline2;

	drawline1.x = outLine.cx - (_inLineParam.range / 2) * outLine.mx;
	drawline1.y = outLine.cy - (_inLineParam.range / 2) * outLine.my;
	drawline2.x = outLine.cx + (_inLineParam.range / 2) * outLine.mx;
	drawline2.y = outLine.cy + (_inLineParam.range / 2) * outLine.my;

	//AddStepRetMarker(TYPE_STEP_OUTPUT, MakeMarker(TYPE_MAKER_LINE1, MAKE_RET_COLOR(255, 0, 0), drawline1.x, drawline1.y, 0, drawline2.x, drawline2.y, ""));

	std::vector<PT32>().swap(fPoints);
	fPoints.clear();

	return true;
}


double hv::v1::algorithm::algorithm::getAlignmentDegreeFromDegree(double _inSrcDegree, double _inBaseDegree) {
	return GetAlignmentDegreeFromDegree(_inSrcDegree, _inBaseDegree);
}