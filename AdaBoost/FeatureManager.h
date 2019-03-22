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
	void setImage(cv::Mat& image,int label,int index);
	void preCacheData(int indexCacheSize);
	void clearCacheData() { indexCacheCount = 0; }
	static float calcDelta(cv::Mat& sum,cv::Mat& sumsq,cv::Rect& area);
	int getFeatureCount() { return allFeatures.size(); }
	void getSortedSample(int featureIdx,cv::Mat& sorted);
	int getSampleCount() { return posCount + negCount; }
	int getLable(int sampleIdx) { return labelMat.at<int>(0, sampleIdx); }
	float getFeatureValue(int fi, int si);
	Feature getFeature(int fi) { return allFeatures[fi]; }
	void getSumMat(int si,cv::Mat& sum);
	cv::Size getWinSize() { return cv::Size(winWidth, winHeight); }
private:
	void genFeatures(int width, int height);
	std::vector<Feature> allFeatures;
	cv::Mat sumMat;
	cv::Mat labelMat;
	cv::Mat deltaMat;
	cv::Mat preIndexCache;
	int winWidth;
	int winHeight;
	int posCount, negCount;
	int indexCacheCount=0;
};