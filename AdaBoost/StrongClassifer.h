#pragma once
#include "WeakClassifer.h"
#include <opencv2/opencv.hpp>
#include "WeakClassifer.h"
#include "FeatureManager.h"

class StrongClassifer
{
public:
	StrongClassifer();
	~StrongClassifer();
	bool train(int posCount,int negCount,FeatureManager& featureManager);
	char predict(int si);
	void updateWeight(cv::Ptr<WeakClassifer> weakClassifer, FeatureManager& featureManager);
	void setMinHitRate(float rate) { minHitRate = rate; }
	void setMaxFARate(float rate) { maxFARate = rate; }
private:
	float weakSum(int si);
	bool isErrDesired();
	std::vector<cv::Ptr<WeakClassifer>> weaks;
	cv::Mat weight;
	float minHitRate=0.99;//最小召回率
	float maxFARate=0.5;//最大误报率
	float threshold = 0;
	int posCount, negCount;
};