#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Feature.h"

using namespace std;
using namespace cv;

//产生所有的feature
vector<Feature> genFeatures(int width, int height)
{
	vector<Feature> result;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++) 
		{
			for (int dx = 1; dx <=width; dx++)
			{
				for (int dy = 1; dy <= height; dy++)
				{
					//A类feature
					if (x+dx*2<=width&&y+dy<=height)
					{
						Feature feature(1,Rect(x,y,2*dx,dy),-2,Rect(x+dx,x,dx,dy));
						result.push_back(feature);
					}
					//B类feature
					if (x+dx<=width && y+2*dy<=height)
					{
						Feature feature(1, Rect(x, y, dx, 2 * dy), -2, Rect(x, y, dx, dy));
						result.push_back(feature);
					}
					//C类feature
					if (x+3*dx<=width && y+dy<=height)
					{
						Feature feature(1,Rect(x,y,3*dx,dy),-3,Rect(x+dx,y,dx,dy));
						result.push_back(feature);
					}
					//D类feature
					if (x+2*dx<=width && y+2*dy<=height)
					{
						Feature feature(1, Rect(x, y, 2 * dx, 2 * dy),
							-2, Rect(x + dx, y, dx, dy),
							-2, Rect(x, y + dy, dx, dy));
						result.push_back(feature);
					}
				}
			}
		}
	}
	return std::move(result);
}

int main() 
{
	//构造测试mat
	Mat testMat(4, 4, CV_8UC1, Scalar(0));
	int sum = 1;
	for (size_t i = 0; i < testMat.rows; i++)
	{
		uchar* row=testMat.ptr<uchar>(i);
		for (size_t j = 0; j < testMat.cols; j++)
		{
			row[j] = sum++;
		}
	}
	vector<Feature> allFeature = genFeatures(testMat.cols, testMat.rows);
	Mat sumMat;
	integral(testMat, sumMat);
	for (auto& feature:allFeature)
	{
		Mat showFeature(4, 4, CV_8UC3);
		feature.draw(showFeature);
		double fv = feature.calc(sumMat);
		cout << "feature value:" << fv << endl;
	}
	cout << "vj boost" << endl;
	return 0;
}