#include "StrongClassifer.h"

using namespace std;
using namespace cv;

StrongClassifer::StrongClassifer()
{
}

StrongClassifer::~StrongClassifer()
{
}

bool StrongClassifer::train(int _posCount, int _negCount, FeatureManager& featureManager)
{
	posCount = _posCount;
	negCount = _negCount;
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
	do{
		cout << "========stong classifer=======" << endl;
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
	} while (!isErrDesired());
	return true;
}

char StrongClassifer::predict(int si)
{
	float sum=weakSum(si);
	if (sum>=threshold)
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
		if (label == featureManager.getLable(si))
		{
			weight.at<double>(0, si) *= weakClassifer->getBeta();
		}
	}
}

float StrongClassifer::weakSum(int si)
{
	float sum = 0;
	for (Ptr<WeakClassifer>& pw : weaks)
	{
		sum += pw->predict(si)*pw->getAlpha();
	}
	return sum;
}

bool StrongClassifer::isErrDesired()
{
	int numPos=0;
	int numPosTrue = 0;
	int numNegTrue = 0;
	int sampleCount = posCount + negCount;
	vector<float> eval(posCount);
	for (int i = 0; i < posCount; i++) {
		eval[numPos++] = weakSum(i);
	}
	std::sort(&eval[0], &eval[0] + numPos);
	int thresholdIdx = (int)((1.0F - minHitRate) * numPos);
	threshold = eval[thresholdIdx];
	numPosTrue = numPos - thresholdIdx;
	for (int i = thresholdIdx - 1; i >= 0; i--)
		if (abs(eval[i] - threshold) < FLT_EPSILON)
			numPosTrue++;
	float hitRate = ((float)numPosTrue) / ((float)numPos);
	cout << "current hitRate: "<< hitRate << endl;
	for (int i = posCount; i < sampleCount; i++)
	{
		if (predict(i)==1) {
			numNegTrue++;
		}
	}
	float falseAlarm = ((float)numNegTrue) / ((float)negCount);
	cout << "current false alarm: " << falseAlarm << endl;
	return falseAlarm <= maxFARate;
}
