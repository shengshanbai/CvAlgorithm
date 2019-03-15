#pragma once
#include<vector>
#include "Feature.h"
#include <opencv2/opencv.hpp>

class FeatureManager
{
public:
	FeatureManager(int winW,int winH);
	~FeatureManager();
	void init(int _posCount,int _negCount);
	void setImage(cv::Mat& image,char label,int index);
	float calcDelta(cv::Mat& sum,cv::Mat& sumsq,cv::Rect& area);
	int getFeatureCount() { return allFeatures.size(); }
	void getSortedSample(int featureIdx,cv::Mat& sorted);
	int getSampleCount() { return posCount + negCount; }
	char getLable(int sampleIdx) { return labelMat.at<char>(0, sampleIdx); }
	float getFeatureValue(int fi, int si);
	Feature getFeature(int fi) { return allFeatures[fi]; }
	void getSumMat(int si,cv::Mat& sum);
private:
	void genFeatures(int width, int height);
	std::vector<Feature> allFeatures;
	cv::Mat sumMat;
	cv::Mat labelMat;
	cv::Mat deltaMat;
	int winWidth;
	int winHeight;
	int posCount, negCount;
};