#include "WeakClassifer.h"
#include <limits>
#include <mutex>

using namespace std;
using namespace cv;

WeakClassifer::WeakClassifer()
{
}

WeakClassifer::~WeakClassifer()
{
}

class PickBestFeature : public ParallelLoopBody {
public:
	PickBestFeature(double& _minErr,
		int& _featureId,
		int& _polarity,
		float& _threshold,
		FeatureManager& _featureManager,
		double _posSumW,
		double _negSumW,
		cv::Mat& _weight,
		mutex& _lock):minErr(_minErr),
		featureId(_featureId),
		polarity(_polarity),
		threshold(_threshold), 
		featureManager(_featureManager),
		posSumW(_posSumW),
		negSumW(_negSumW),
		weight(_weight),
		lock(_lock){}

	void operator()(const Range& range) const
	{
		int sampleCount = featureManager.getSampleCount();
		for (int fi = range.start; fi < range.end; ++fi)
		{
			Mat sortedSample;
			featureManager.getSortedSample(fi, sortedSample);
			//Ѱ������зֵ�
			double posAcc = 0, negAcc = 0;
			for (int i = 0; i < sampleCount; i++)
			{
				ushort index = sortedSample.at<ushort>(0, i);
				double lErr = posAcc + (negSumW - negAcc);
				double rErr = negAcc + (posSumW - posAcc);
				if (lErr<rErr)
				{
					std::lock_guard<std::mutex> lck(lock);
					if (lErr<minErr)
					{
						minErr = lErr;
						polarity = -1;
						threshold = featureManager.getFeatureValue(fi, index);
						featureId = fi;
					}
				}
				else
				{
					std::lock_guard<std::mutex> lck(lock);
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
					posAcc += weight.at<double>(0, index);
				}
				else
				{
					negAcc += weight.at<double>(0, index);
				}
			}
		}
	}

	double& minErr;
	int& featureId;
	int& polarity;
	float& threshold;
	FeatureManager & featureManager;
	double posSumW;
	double negSumW;
	cv::Mat& weight;
	mutex& lock;
};

bool WeakClassifer::train(double posSumW, double negSumW, cv::Mat & weight, FeatureManager & featureManager)
{
	pfmanager = &featureManager;
	int featureCount = featureManager.getFeatureCount();
	//��ÿ��feature���д���
	double minErr = std::numeric_limits<double>::max();
	std::mutex lock;
	parallel_for_(Range(0, featureCount), PickBestFeature(minErr,featureId,
		polarity,threshold,featureManager,posSumW,negSumW,weight,lock));
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
