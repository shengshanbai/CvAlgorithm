#pragma once
#include <opencv2/opencv.hpp>

#define MAX_RECT_COUNT 3

class Feature
{
public:
	Feature();
	Feature(float weight0, cv::Rect2f& rect0, float weight1, cv::Rect2f& rect1,
		float weight2 = 0,cv::Rect2f rect2=cv::Rect2f());
	~Feature();
	//画出feature用于测试
	void draw(cv::Mat& image);
	double calc(cv::Mat& sum) const;
	//计算在指定区域的feature值
	double computeNormalFeature(cv::Mat& sum, cv::Mat& sumSq, cv::Rect2f detect);
	void save(cv::FileStorage& fs);
	void load(cv::FileNode& node);

	struct WeightRect
	{
		float weight;
		cv::Rect2f r;
	} rect[MAX_RECT_COUNT];
private:

};

Feature operator* (float x, const Feature& y);

Feature operator* (const Feature& y, float x);