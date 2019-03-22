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
	int predict(int si);
	int predict(cv::Mat& sum, cv::Mat& sumSq, cv::Rect2f detect, float sizeScale);
	void updateWeight(cv::Ptr<WeakClassifer> weakClassifer, FeatureManager& featureManager);
	void setMinHitRate(float rate) { minHitRate = rate; }
	void setMaxFARate(float rate) { maxFARate = rate; }
	void save(cv::FileStorage& fs);
	void load(cv::FileNode& fnode);
private:
	float weakSum(int si);
	bool isErrDesired(FeatureManager& featureManager);
	std::vector<cv::Ptr<WeakClassifer>> weaks;
	cv::Mat weight;
	float minHitRate=0.994;//最小召回率
	float maxFARate=0.5;//最大误报率
	float threshold = 0;
	int posCount, negCount;
};