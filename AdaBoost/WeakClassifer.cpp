#include "WeakClassifer.h"
#include <limits>
using namespace std;
using namespace cv;

WeakClassifer::WeakClassifer()
{
}

WeakClassifer::~WeakClassifer()
{
}

bool WeakClassifer::train(double posSumW, double negSumW, cv::Mat & weight, FeatureManager & featureManager)
{
	pfmanager = &featureManager;
	int featureCount = featureManager.getFeatureCount();
	int sampleCount = featureManager.getSampleCount();
	//对每个feature进行处理
	double minErr = std::numeric_limits<double>::max();
	for (int fi = 0; fi < featureCount; fi++)
	{
		Mat sortedSample;
		featureManager.getSortedSample(fi, sortedSample);
		//寻找最佳切分点
		double posAcc = 0, negAcc = 0;
		for (int i = 0; i < sampleCount; i++)
		{
			ushort index = sortedSample.at<ushort>(0, i);
			double lErr = posAcc + (negSumW-negAcc);
			double rErr = negAcc + (posSumW - posAcc);
			if (lErr<rErr)
			{
				if(lErr<minErr)
				{
					minErr = lErr;
					polarity = -1;
					threshold = featureManager.getFeatureValue(fi, index);
					featureId = fi;
				}
			}
			else
			{
				if (rErr<minErr)
				{
					minErr = rErr;
					polarity = 1;
					threshold = featureManager.getFeatureValue(fi, index);
					featureId = fi;
				}
			}
			char label = featureManager.getLable(index);
			if (label == 1)
			{
				posAcc+=weight.at<double>(0,index);
			}
			else
			{
				negAcc+= weight.at<double>(0, index);
			}
		}
	}
	cout << "the minErr:" << minErr << endl;
	feature = featureManager.getFeature(featureId);
	beta = minErr / (1 - minErr);
	alpha = log(1 / beta);
	return true;
}

char WeakClassifer::predict(int sampleIdx)
{
	Mat sum;
	float fvalue=pfmanager->getFeatureValue(featureId,sampleIdx);
	if (polarity*fvalue < polarity*threshold)
	{
		return 1;
	}
	return 0;
}
