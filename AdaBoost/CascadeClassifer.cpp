#include "CascadeClassifer.h"
#include <cmath>
using namespace std;
using namespace cv;

CascadeClassifer::CascadeClassifer()
{
}

CascadeClassifer::~CascadeClassifer()
{

}

bool CascadeClassifer::train(int stage, CvCascadeImageReader & imageReader, FeatureManager& fmanager, int _posCount, int _negCount, float requirAF,std::string modelDir)
{
	winSize = fmanager.getWinSize();
	saveStage(modelDir,stage);
	posCount = _posCount;
	negCount = _negCount;
	StrongClassifer strongClassifer;
	fmanager.init(posCount, negCount);
	for (int i = 0; i < stage; i++)
	{
		cout << endl;
		cout << "<begin train stage:"<<i<<"**********"<< endl;
		if (!updateTrainSet(imageReader, fmanager, requirAF)) {
			break;
		}
		fmanager.preCacheData(1536);
		Ptr<StrongClassifer> classifer = makePtr<StrongClassifer>();
		classifer->train(posCount, negCount, fmanager);
		classifers.push_back(classifer);
		cout << "********** end train stage>"<< endl;
		saveClassifer(modelDir, i, classifer);
	}
	return true;
}

int CascadeClassifer::predict(int si)
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

void CascadeClassifer::load(std::string modelDir)
{
	string filename = modelDir + "/" + MFILE_NAME;
	FileStorage fs(filename, FileStorage::READ);
	FileNode node=fs.getFirstTopLevelNode();
	int stage;
	node["stage"] >> stage;
	node["winSize"] >> winSize;
	fs.release();
	classifers.reserve(stage);
	for (int i = 0; i < stage; i++)
	{
		string filename = modelDir + "/" + MSTRONG_NAME + to_string(i) + ".xml";
		FileStorage fs(filename, FileStorage::READ);
		FileNode fnode = fs.getFirstTopLevelNode();
		Ptr<StrongClassifer> classifer = makePtr<StrongClassifer>();
		classifer->load(fnode);
		classifers.push_back(classifer);
	}
}

void CascadeClassifer::saveStage(std::string modelDir, int stage)
{
	string filename = modelDir + "/" + MFILE_NAME;
	FileStorage fs(filename, FileStorage::WRITE);
	fs << FileStorage::getDefaultObjectName(MFILE_NAME) << "{";
	fs << "stage" << stage;
	fs << "winSize" << winSize;
	fs << "}";
	fs.release();
}

void CascadeClassifer::saveClassifer(std::string modelDir, int index, cv::Ptr<StrongClassifer> classifer)
{
	string filename = modelDir + "/"+MSTRONG_NAME + to_string(index) + ".xml";
	FileStorage fs(filename, FileStorage::WRITE);
	fs << FileStorage::getDefaultObjectName(filename) << "{";
	classifer->save(fs);
	fs << "}";
	fs.release();
}

bool CascadeClassifer::updateTrainSet(CvCascadeImageReader & imageReader, FeatureManager & fmanager, float requirAF)
{
	imageReader.restart();
	fmanager.clearCacheData();
	int readed = 0;
	int consumed = 0;
	int posGet=fillPassedSample(imageReader, fmanager, true, readed, consumed);
	if (posGet!=posCount)
	{
		cout << "pos sample is not enought" << endl;
		return false;
	}
	cout << "pos count:" << posGet << " consumed:" << consumed << endl;
	int negGet=fillPassedSample(imageReader, fmanager, false, readed, consumed);
	if ((float)(negGet + 1) / consumed <= requirAF)
	{
		cout << "achieve the required accuracy:" << (float)(negGet + 1) / consumed << endl;
		return false;
	}
	if (negGet!=negCount)
	{
		cout << "neg sample is not enought" << endl;
		return false;
	}
	cout << "neg count:" << negGet << " consumed:" << consumed << endl;
	return true;
}

int CascadeClassifer::fillPassedSample(CvCascadeImageReader & imageReader, FeatureManager & fmanager, bool pos, int& readed, int& consumed)
{
	int index = 0;
	int target = pos ? posCount : negCount;
	int getCount = 0;
	consumed = 0;
	while (getCount<target)
	{
		Mat img(imageReader.winSize, CV_8UC1);
		if (pos)
		{
			imageReader.getPos(img);
		}
		else
		{
			imageReader.getNeg(img);
		}
		if (img.empty())
		{
			break;
		}
		index++;
		consumed++;
		fmanager.setImage(img, pos?1:0, readed);
		if (predict(readed)==1)
		{
			getCount++;
			readed++;
		}
	}
	return getCount;
}

bool CascadeClassifer::predict(cv::Mat& gray,std::vector<cv::Rect>& faces)
{
	int imageW = gray.cols;
	int imageH = gray.rows;
	Mat sum;
	Mat sumSq;
	cv::integral(gray, sum, sumSq);
	for (int i = 0;; i++)
	{
		float sizeScale=pow(scaleFactor,i);
		float width = winSize.width*sizeScale;
		float height = winSize.height*sizeScale;
		if (width>imageW||height>imageH)
		{
			break;
		}
		for (int x = 0; x < imageW; x+=step)
		{
			for (int y = 0; y < imageH; y+=step)
			{
				if (x + width < imageW && y + height < imageH)
				{
					Rect2f rect(x, y, width, height);
					if (predict(sum, sumSq, rect, sizeScale) == 1)
					{
						faces.push_back(rect);
					}
				}
			}
		}
	}
	return !faces.empty();
}

int CascadeClassifer::predict(cv::Mat & sum, cv::Mat & sumSq, cv::Rect2f detect, float sizeScale)
{
	for (cv::Ptr<StrongClassifer>& pStrong : classifers)
	{
		if (pStrong->predict(sum,sumSq,detect,sizeScale) == 0)
		{
			return 0;
		}
	}
	return 1;
}
