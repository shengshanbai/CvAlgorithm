#include "CascadeClassifer.h"
using namespace std;
using namespace cv;

CascadeClassifer::CascadeClassifer()
{
}

CascadeClassifer::~CascadeClassifer()
{
}

bool CascadeClassifer::train(int stage,ImageReader & imageReader, FeatureManager& fmanager, int _posCount, int _negCount)
{
	posCount = _posCount;
	negCount = _negCount;
	StrongClassifer strongClassifer;
	fmanager.init(posCount, negCount);
	for (int i = 0; i < stage; i++)
	{
		updateTrainSet(imageReader, fmanager);
		fmanager.preCacheData(1536);
		Ptr<StrongClassifer> classifer = makePtr<StrongClassifer>();
		classifer->train(posCount, negCount, fmanager);
		classifers.push_back(classifer);
	}
	return true;
}

char CascadeClassifer::predict(int si)
{
	for (cv::Ptr<StrongClassifer>& pStrong:classifers)
	{
		if (pStrong->predict(si)==0)
		{
			return 0;
		}
	}
	return 1;
}

void CascadeClassifer::updateTrainSet(ImageReader & imageReader, FeatureManager & fmanager)
{
	fmanager.clearCacheData();
	int readed = 0;
	fillPassedSample(imageReader, fmanager, true, readed);
	fillPassedSample(imageReader, fmanager, false, readed);
}

int CascadeClassifer::fillPassedSample(ImageReader & imageReader, FeatureManager & fmanager, bool pos, int& readed)
{
	int index = 0;
	int target = pos ? posCount : negCount;
	int getCount = 0;
	while (getCount<target)
	{
		Mat img;
		img = pos?imageReader.readPosImage(index):
			imageReader.readNegImage(index);
		if (img.empty())
		{
			cout << "the " << (pos ? "POS" : "NEG") << " image not enough to train." << endl;
			exit(1);
		}
		index++;
		fmanager.setImage(img, pos?1:0, readed);
		if (predict(getCount)==1)
		{
			getCount++;
			readed++;
		}
	}
	return 0;
}
