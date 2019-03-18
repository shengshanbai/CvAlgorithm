#pragma once
#include "ImageReader.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "StrongClassifer.h"
#include "FeatureManager.h"

class CascadeClassifer
{
public:
	CascadeClassifer();
	~CascadeClassifer();
	bool train(int stage,ImageReader& imageReader, FeatureManager& fmanager,int _posCount, int _negCount);
	char predict(int si);
private:
	void updateTrainSet(ImageReader& imageReader,FeatureManager& fmanager);
	int fillPassedSample(ImageReader& imageReader, FeatureManager& fmanager,bool pos,int& readed);
	std::vector<cv::Ptr<StrongClassifer>> classifers;
	int posCount;
	int negCount;
};