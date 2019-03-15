#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class  ImageReader
{
public:
	 ImageReader(const std::string& posDir,const std::string negDir);
	~ ImageReader();
	cv::Mat readPosImage(int& posIndex);
	cv::Mat readNegImage(int& negIndex);
private:
	void scanImageFile(const std::string& dir,std::vector<std::string>& files);
	std::vector<std::string> posFiles;
	std::vector<std::string> negFiles;
};