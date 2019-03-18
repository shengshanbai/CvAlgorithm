#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Feature.h"
#include "ImageReader.h"
#include "CascadeClassifer.h"
#include "FeatureManager.h"

using namespace std;
using namespace cv;

#define POSDIR "E:\\Projects\\Face-Detection\\faces"
#define NEGDIR "E:\\Projects\\Face-Detection\\nonfaces"
#define WIN_W 20
#define WIN_H 20
#define POSCOUNT 500
#define NEGCOUNT 500

int main()
{
	setNumThreads(8);
	ImageReader reader(POSDIR, NEGDIR);
	FeatureManager featureManager(WIN_W, WIN_H);
	CascadeClassifer classifer;
	classifer.train(2, reader, featureManager,POSCOUNT,NEGCOUNT);
	int negCout = 0;
	for (int i = 0; i < POSCOUNT+NEGCOUNT; i++)
	{
		char label = classifer.predict(i);
		if (label!=featureManager.getLable(i))
		{
			negCout++;
		}
	}
	cout << "the strong err ratio:" << (float)negCout/(POSCOUNT + NEGCOUNT);
	char temp;
	cin >> temp;
	return 0;
}