#pragma once
#include <opencv2/opencv.hpp>

#define MAX_RECT_COUNT 3

class Feature
{
public:
	Feature();
	Feature(float weight0, cv::Rect& rect0, float weight1, cv::Rect& rect1,
		float weight2 = 0,cv::Rect rect2=cv::Rect());
	~Feature();
	//ª≠≥ˆfeature”√”⁄≤‚ ‘
	void draw(cv::Mat& image);
	double calc(cv::Mat& sum);
private:
	struct WeightRect
	{
		float weight;
		cv::Rect r;
	} rect[MAX_RECT_COUNT];                                                                                   
};