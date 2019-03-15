#include "StrongClassifer.h"

using namespace std;
using namespace cv;

StrongClassifer::StrongClassifer()
{
}

StrongClassifer::~StrongClassifer()
{
}

bool StrongClassifer::train(int stage, int posCount, int negCount, FeatureManager& featureManager)
{
	int sampleCount = posCount + negCount;
	weight.create(1, sampleCount, CV_64FC1);
	for (int i = 0; i < posCount; i++)
	{
		weight.at<double>(0, i) = 1.0 / (2 * posCount);
	}

	for (int i = posCount; i < sampleCount; i++)
	{
		weight.at<double>(0, i) = 1.0 / (2 * negCount);
	}
	for (int i = 0; i < stage; i++)
	{
		normalize(weight, weight, 1,0, NORM_L1);
		double posSumW=0, negSumW=0;
		for (int i = 0; i < sampleCount; i++)
		{
			if (i < posCount)
				posSumW += weight.at<double>(0, i);
			else
				negSumW+= weight.at<double>(0, i);
		}
		Ptr<WeakClassifer> weakClassifer = makePtr<WeakClassifer>();
		weakClassifer->train(posSumW, negSumW, weight, featureManager);
		weaks.push_back(weakClassifer);
		updateWeight(weakClassifer, featureManager);
	}
	return true;
}

char StrongClassifer::predict(int si)
{
	float sum=0;
	float sumAlpha=0;
	for (Ptr<WeakClassifer> pw:weaks)
	{
		sum += pw->predict(si)*pw->getAlpha();
		sumAlpha += pw->getAlpha();
	}
	if (sum>=0.5*sumAlpha)
	{
		return 1;
	}
	return 0;
}

void StrongClassifer::updateWeight(Ptr<WeakClassifer> weakClassifer, FeatureManager& featureManager)
{
	int sampleCount = featureManager.getSampleCount();
	for (int si = 0; si < sampleCount; si++)
	{
		char label = predict(si);
		if (label!= featureManager.getLable(si))
		{
			weight.at<double>(0, si) *= weakClassifer->getBeta();
		}
	}
}
