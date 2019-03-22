#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Feature.h"
#include "imagestorage.h"
#include "CascadeClassifer.h"
#include "FeatureManager.h"

using namespace std;
using namespace cv;

#define FACE_VEC "E:\\dataset\\boostImages\\face.vec"
#define NON_FACE_INFO "E:\\dataset\\boostImages\\nface.txt"
#define MODEL_DIR "E:\\Projects\\CvAlgorithm\\AdaBoost\\model"
#define TEST_FILE "E:\\Projects\\FaceDetection\\FaceDetection\\Test\\2.jpg"

#define WIN_W 19
#define WIN_H 19
#define POSCOUNT 4000
#define NEGCOUNT 4000

int train() 
{
	CvCascadeImageReader imgReader;
	if (!imgReader.create(FACE_VEC, NON_FACE_INFO, Size(WIN_W, WIN_H)))
	{
		cout << "Image reader can not be created from vec " << FACE_VEC
			<< " and bg " << NON_FACE_INFO << "." << endl;
		return 2;
	}
	FeatureManager featureManager(WIN_W, WIN_H);
	CascadeClassifer classifer;
	classifer.train(20, imgReader, featureManager, POSCOUNT, NEGCOUNT, 0.0001f, MODEL_DIR);
	return 0;
}

int main()
{
	setNumThreads(8);
	//train();
	CascadeClassifer classifer;
	classifer.load(MODEL_DIR);
	Mat image = imread(TEST_FILE);
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	vector<Rect> faces;
	bool detect=classifer.predict(gray, faces);
	if (detect)
	{
		for (auto& face : faces)
		{
			rectangle(image, face, Scalar(0, 0, 255));
		}
	}
	imshow("face", image);
	waitKey();
	destroyAllWindows();
	return 0;
}