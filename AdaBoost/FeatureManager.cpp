#include "FeatureManager.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

FeatureManager::FeatureManager(int winW, int winH)
{
	genFeatures(winW, winH);
	winWidth = winW;
	winHeight = winH;
}

FeatureManager::~FeatureManager()
{
}

void FeatureManager::init(int _posCount, int _negCount)
{
	posCount = _posCount;
	negCount = _negCount;
	int sampleCount = posCount + negCount;
	labelMat.create(1,sampleCount,CV_8SC1);
	sumMat.create(sampleCount,(winWidth+1)*(winHeight+1),CV_32SC1);
	deltaMat.create(1,sampleCount,CV_32FC1);
}

void FeatureManager::setImage(cv::Mat & image,char label,int index)
{
	labelMat.at<char>(0, index) = label;
	Mat sum(winHeight + 1, winWidth + 1,CV_32SC1,sumMat.ptr<int>(index));
	Mat sumSq;
	cv::integral(image, sum, sumSq);
	deltaMat.at<float>(0, index) = calcDelta(sum, sumSq, Rect(0,0,winWidth, winHeight));
}

//����feature����
class FSort
{
public:
	FSort(cv::AutoBuffer<float>& _fbuf) :fbuf(_fbuf)
	{
	}
	bool operator()(ushort& a, ushort& b) const {
		return fbuf[a] < fbuf[b];
	}
	cv::AutoBuffer<float>& fbuf;
};


class PreIndexSort : public ParallelLoopBody {
public:
	PreIndexSort(cv::Mat& _sortedIndex,int _sampleCount,
		std::vector<Feature>& _allFeatures,
		cv::Mat& _sumMat,
		cv::Mat& _deltaMat,
		int _winWidth,
		int _winHeight):sortedIndex(_sortedIndex),
		allFeatures(_allFeatures),
		sampleCount(_sampleCount),
		sumMat(_sumMat),
		deltaMat(_deltaMat),
		winWidth(_winWidth),
		winHeight(_winHeight){}
	void operator()(const Range& range) const
	{
		for (int fi = range.start; fi < range.end; ++fi)
		{
			ushort* rowData = sortedIndex.ptr<ushort>(fi);
			for (ushort i = 0; i < sampleCount; i++)
			{
				rowData[i] = i;
			}
			const Feature& feature = allFeatures[fi];
			cv::AutoBuffer<float> fValueBuf(sampleCount);
			//��������sample��feature
			for (int si = 0; si < sampleCount; si++)
			{
				cv::Mat sum(winHeight + 1, winWidth + 1, CV_32SC1, sumMat.ptr<int>(si));
				float fValue = feature.calc(sum) / deltaMat.at<float>(0, si);
				fValueBuf[si] = fValue;
			}
			std::sort(rowData, rowData + sampleCount, FSort(fValueBuf));
		}
	}
	cv::Mat& sortedIndex;
	int sampleCount;
	std::vector<Feature> allFeatures;
	cv::Mat& sumMat;
	cv::Mat& deltaMat;
	int winWidth;
	int winHeight;
};

/*
��ǰ����û������ݣ����Լ��ټ���
indexCacheSize:���������cache��С����λ��MB��
*/
void FeatureManager::preCacheData(int indexCacheSize)
{
	long cacheByte = indexCacheSize * 1024 * 1024;
	int sampleCount = posCount + negCount;
	long rowSize = sampleCount * sizeof(ushort);
	int featureCount = getFeatureCount();
	indexCacheCount = cacheByte / rowSize > featureCount ? featureCount : cacheByte / rowSize;
	preIndexCache.create(indexCacheCount, sampleCount, CV_16UC1);
	parallel_for_(Range(0, indexCacheCount), PreIndexSort(preIndexCache, sampleCount, allFeatures, sumMat, deltaMat, winWidth, winHeight));
}

//���ݻ���ͼ����ָ�������deltaֵ
float FeatureManager::calcDelta(cv::Mat & sum, cv::Mat & sumsq, cv::Rect & area)
{
	double sumV = sum.at<int>(area.y, area.x) + sum.at<int>(area.y + area.height, area.x + area.width)
		- sum.at<int>(area.y, area.x + area.width) - sum.at<int>(area.y + area.height, area.x);
	double sumSq= sumsq.at<double>(area.y, area.x) + sumsq.at<double>(area.y + area.height, area.x + area.width)
		- sumsq.at<double>(area.y, area.x + area.width) - sumsq.at<double>(area.y + area.height, area.x);
	int N = area.width*area.height;
	return sqrt(N*sumSq - sumV * sumV) / N;
}

void FeatureManager::getSortedSample(int featureIdx, cv::Mat & sorted)
{
	//ʹ�û���
	if (featureIdx<indexCacheCount)
	{
		sorted = preIndexCache.row(featureIdx);
		return;
	}
	int sampleCount = posCount + negCount;
	sorted.create(1, sampleCount, CV_16UC1);
	for (ushort i = 0; i < sampleCount; i++)
	{
		sorted.at<ushort>(0, i) = i;
	}
	Feature& feature = allFeatures[featureIdx];
	cv::AutoBuffer<float> fValueBuf(sampleCount);
	//��������sample��feature
	for (int si = 0; si < sampleCount; si++)
	{
		cv::Mat sum(winHeight + 1, winWidth + 1, CV_32SC1, sumMat.ptr<int>(si));
		float fValue = feature.calc(sum) / deltaMat.at<float>(0,si);
		fValueBuf[si] = fValue;
	}
	std::sort(sorted.ptr<ushort>(0), sorted.ptr<ushort>(0) + sampleCount,FSort(fValueBuf));
}

float FeatureManager::getFeatureValue(int fi, int si)
{
	Feature& feature = allFeatures[fi];
	cv::Mat sum(winHeight + 1, winWidth + 1, CV_32SC1, sumMat.ptr<int>(si));
	float fValue = feature.calc(sum) / deltaMat.at<float>(0, si);
	return fValue;
}

void FeatureManager::getSumMat(int si,cv::Mat& sum)
{
	sum = Mat(winHeight + 1, winWidth + 1, CV_32SC1, sumMat.ptr<int>(si));
}

//�������е�feature
void FeatureManager::genFeatures(int width, int height)
{
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int dx = 1; dx <= width; dx++)
			{
				for (int dy = 1; dy <= height; dy++)
				{
					//A��feature
					if (x + dx * 2 <= width && y + dy <= height)
					{
						Feature feature(1, Rect(x, y, 2 * dx, dy), -2, Rect(x + dx, y, dx, dy));
						allFeatures.push_back(std::move(feature));
					}
					//B��feature
					if (x + dx <= width && y + 2 * dy <= height)
					{
						Feature feature(1, Rect(x, y, dx, 2 * dy), -2, Rect(x, y, dx, dy));
						allFeatures.push_back(std::move(feature));
					}
					//C��feature
					if (x + 3 * dx <= width && y + dy <= height)
					{
						Feature feature(1, Rect(x, y, 3 * dx, dy), -3, Rect(x + dx, y, dx, dy));
						allFeatures.push_back(std::move(feature));
					}
					//D��feature
					if (x + 2 * dx <= width && y + 2 * dy <= height)
					{
						Feature feature(1, Rect(x, y, 2 * dx, 2 * dy),
							-2, Rect(x + dx, y, dx, dy),
							-2, Rect(x, y + dy, dx, dy));
						allFeatures.push_back(std::move(feature));
					}
				}
			}
		}
	}
}