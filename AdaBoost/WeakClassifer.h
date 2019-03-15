#pragma once
#include "FeatureManager.h"
#include "Feature.h"

class WeakClassifer
{
public:
	WeakClassifer();
	~WeakClassifer();
	bool train(double posSumW,double negSumW,cv::Mat& weight,
		FeatureManager& featureManager);
	char predict(int sampleIdx);
	float getAlpha() { return alpha; }
	float getBeta() { return beta; }
private:
	int polarity;
	float threshold;
	Feature feature;
	FeatureManager* pfmanager;
	int featureId = 0;
	float beta;
	float alpha;
};