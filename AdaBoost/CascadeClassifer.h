#pragma once
#include "imagestorage.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "StrongClassifer.h"
#include "FeatureManager.h"
#include <string>

#define MFILE_NAME "cascade.xml"
#define MSTRONG_NAME "strong"

class CascadeClassifer
{
public:
	CascadeClassifer();
	~CascadeClassifer();
	bool train(int stage, CvCascadeImageReader& imageReader, FeatureManager& fmanager,
		int _posCount, int _negCount,float requirAF,std::string modelDir);
	int predict(int si);
	bool predict(cv::Mat& gray,std::vector<cv::Rect>& faces);
	int predict(cv::Mat& sum,cv::Mat& sumSq,cv::Rect2f detect,float sizeScale);
	void load(std::string modelDir);
private:
	void saveStage(std::string modelDir,int stage);
	void saveClassifer(std::string modelDir, int index, cv::Ptr<StrongClassifer> classifer);
	bool updateTrainSet(CvCascadeImageReader& imageReader,FeatureManager& fmanager,float requirAF);
	int fillPassedSample(CvCascadeImageReader& imageReader, FeatureManager& fmanager,bool pos,int& readed,int& consumed);
	std::vector<cv::Ptr<StrongClassifer>> classifers;
	int posCount;
	int negCount;
	cv::Size winSize;
	float scaleFactor = 2;
	float step = 1;
};