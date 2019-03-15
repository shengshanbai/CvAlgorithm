#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Feature.h"
#include "ImageReader.h"
#include "StrongClassifer.h"
#include "FeatureManager.h"

using namespace std;
using namespace cv;

#define POSDIR "E:\\Projects\\Face-Detection\\faces"
#define NEGDIR "E:\\Projects\\Face-Detection\\nonfaces"
#define WIN_W 20
#define WIN_H 20
#define POSCOUNT 100
#define NEGCOUNT 100

int main()
{
	ImageReader reader(POSDIR, NEGDIR);
	StrongClassifer strongClassifer;
	FeatureManager featureManager(WIN_W, WIN_H);
	featureManager.init(POSCOUNT, NEGCOUNT);
	int index = 0;
	for (int i = 0; i < POSCOUNT;i++)
	{
		Mat posMat = reader.readPosImage(i);
		featureManager.setImage(posMat, 1, index);
		index++;
	}
	for (int i = 0; i < NEGCOUNT;i++)
	{
		Mat negMat = reader.readNegImage(i);
		featureManager.setImage(negMat,0, index);
		index++;
	}
	strongClassifer.train(2, POSCOUNT, NEGCOUNT,featureManager);
	int negCout = 0;
	for (int i = 0; i < POSCOUNT+NEGCOUNT; i++)
	{
		char label = strongClassifer.predict(i);
		if (label!=featureManager.getLable(i))
		{
			negCout++;
		}
	}
	cout << "the strong err ratio:" << (float)negCout/(POSCOUNT + NEGCOUNT);
	return 0;
}