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
	int predict(int sampleIdx);
	int predict(cv::Mat& sum, cv::Mat& sumSq, cv::Rect2f detect, float sizeScale);
	float getAlpha() { return alpha; }
	float getBeta() { return beta; }
	void save(cv::FileStorage& fs);
	void load(cv::FileNode& fnode);
private:
	int polarity;
	float threshold;
	Feature feature;
	FeatureManager* pfmanager;
	int featureId = 0;
	float beta;
	float alpha;
};