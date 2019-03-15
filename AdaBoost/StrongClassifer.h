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
	bool train(int stage,int posCount,int negCount,FeatureManager& featureManager);
	char predict(int si);
	void updateWeight(cv::Ptr<WeakClassifer> weakClassifer, FeatureManager& featureManager);
private:
	std::vector<cv::Ptr<WeakClassifer>> weaks;
	cv::Mat weight;
};