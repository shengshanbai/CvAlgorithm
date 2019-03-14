#include "Feature.h"
using namespace cv;

Feature::Feature()
{
}

Feature::~Feature()
{
}

Feature::Feature(float weight0, cv::Rect& rect0, float weight1, cv::Rect& rect1,
	float weight2, cv::Rect rect2)
{
	rect[0].weight = weight0;
	rect[0].r = rect0;
	rect[1].weight = weight1;
	rect[1].r = rect1;
	rect[2].weight = weight2;
	rect[2].r = rect2;
}

void Feature::draw(cv::Mat& image)
{
	for (struct WeightRect& wRect:rect)
	{
		if (wRect.weight>0)
		{
			rectangle(image, wRect.r, Scalar(255, 255, 255));
		}
		else if (wRect.weight<0)
		{
			rectangle(image, wRect.r, Scalar(0, 0, 255));
		}
	}
}

double Feature::calc(cv::Mat& sum)
{
	double result = 0;
	for (WeightRect& wRect:rect)
	{
		if (wRect.weight!=0)
		{
			result += wRect.weight*(sum.at<int>(wRect.r.y, wRect.r.x) + sum.at<int>(wRect.r.y + wRect.r.height, wRect.r.x + wRect.r.width) - sum.at<int>(wRect.r.y + wRect.r.height, wRect.r.x) - sum.at<int>(wRect.r.y, wRect.r.x + wRect.r.width));
		}
	}
	return result;
}