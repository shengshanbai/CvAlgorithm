#include "ImageReader.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>

using namespace std;
using namespace cv;

ImageReader::ImageReader(const std::string& posDir, const std::string negDir)
{
	if (!cv::utils::fs::exists(posDir))
	{
		cout << "not exist dir:" << posDir;
		return;
	}
	if (!cv::utils::fs::exists(negDir))
	{
		cout << "not exist dir:" << posDir;
		return;
	}
	scanImageFile(posDir,posFiles);
	scanImageFile(negDir,negFiles);
}

ImageReader::~ImageReader()
{
}

cv::Mat ImageReader::readPosImage(int & posIndex)
{
	cv::Mat image;
	if (posIndex>=0 && posIndex < posFiles.size())
	{
		string& path = posFiles[posIndex];
		image = imread(path,IMREAD_GRAYSCALE);
	}
	return image;
}

cv::Mat ImageReader::readNegImage(int & negIndex)
{
	cv::Mat image;
	if (negIndex >= 0 && negIndex < posFiles.size())
	{
		string& path = negFiles[negIndex];
		image = imread(path, IMREAD_GRAYSCALE);
	}
	return image;
}

void ImageReader::scanImageFile(const std::string & dir, std::vector<string>& files)
{
	cv::utils::fs::glob(dir, "", files);
}
