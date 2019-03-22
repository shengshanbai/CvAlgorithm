#include "Feature.h"
#include "FeatureManager.h"
#include <climits>

using namespace cv;
using namespace std;

Feature::Feature()
{
}

Feature::~Feature()
{
}

Feature::Feature(float weight0, cv::Rect2f& rect0, float weight1, cv::Rect2f& rect1,
	float weight2, cv::Rect2f rect2)
{
	rect[0].weight = weight0;
	rect[0].r = rect0;
	rect[1].weight = weight1;
	rect[1].r = rect1;
	rect[2].weight = weight2;
	rect[2].r = rect2;
}

void Feature::draw(cv::Mat& image)
{
	for (struct WeightRect& wRect:rect)
	{
		if (wRect.weight>0)
		{
			rectangle(image, wRect.r, Scalar(255, 255, 255));
		}
		else if (wRect.weight<0)
		{
			rectangle(image, wRect.r, Scalar(0, 0, 255));
		}
	}
}

double Feature::calc(cv::Mat& sum) const
{
	double result = 0;
	for (const WeightRect& wRect : rect)
	{
		if (wRect.weight != 0)
		{
			result += wRect.weight*(sum.at<int>(wRect.r.y, wRect.r.x) + sum.at<int>(wRect.r.y + wRect.r.height, wRect.r.x + wRect.r.width) - sum.at<int>(wRect.r.y + wRect.r.height, wRect.r.x) - sum.at<int>(wRect.r.y, wRect.r.x + wRect.r.width));
		}
	}
	return result;
}

double Feature::computeNormalFeature(cv::Mat & sum, cv::Mat & sumSq, cv::Rect2f detect)
{
	//转换rect到绝对坐标
	struct WeightRect rectAbs[MAX_RECT_COUNT];
	for (int i = 0; i < MAX_RECT_COUNT; i++)
	{
		rectAbs[i].weight = rect[i].weight;
		rectAbs[i].r.x = rect[i].r.x + detect.x;
		rectAbs[i].r.y = rect[i].r.y + detect.y;
		rectAbs[i].r.width = rect[i].r.width;
		rectAbs[i].r.height = rect[i].r.height;
	}
	double featureValue = 0;
	for (const WeightRect& wRect : rectAbs)
	{
		if (wRect.weight != 0)
		{
			//左上角带小数的sum值
			int topLeftRow = int(wRect.r.y);
			int topLeftCol = int(wRect.r.x);
			float ratio = (wRect.r.y - topLeftRow)*(wRect.r.x - topLeftCol);
			int addOneRow = topLeftRow + 1 > sum.rows - 1 ? sum.rows - 1 : topLeftRow + 1;
			int addOneCol = topLeftCol + 1 > sum.cols - 1 ? sum.cols - 1 : topLeftCol + 1;
			float topLeftValue = ratio * sum.at<int>(addOneRow, addOneCol) + (1 - ratio)*sum.at<int>(topLeftRow, topLeftCol);
			//右下角带小数的sum值
			int bottomRightRow = int(wRect.r.y + wRect.r.height);
			int bottomRightCol = int(wRect.r.x + wRect.r.width);
			ratio = (wRect.r.y + wRect.r.height - bottomRightRow)*(wRect.r.x + wRect.r.width- bottomRightCol);
			addOneRow = bottomRightRow + 1 > sum.rows - 1 ? sum.rows - 1 : bottomRightRow + 1;
			addOneCol = bottomRightCol + 1 > sum.cols - 1 ? sum.cols - 1 : bottomRightCol + 1;
			float bottomRightValue = ratio * sum.at<int>(addOneRow, addOneCol) + (1 - ratio)*sum.at<int>(bottomRightRow, bottomRightCol);
			//左下角带小数的sum值
			float bottomLeftRow = int(wRect.r.y + wRect.r.height);
			float bottomLeftCol = int(wRect.r.x);
			ratio = (wRect.r.x - bottomLeftCol)*(wRect.r.y + wRect.r.height- bottomLeftRow);
			addOneRow = bottomLeftRow + 1 > sum.rows - 1 ? sum.rows - 1 : bottomLeftRow + 1;
			addOneCol = bottomLeftCol + 1 > sum.cols - 1 ? sum.cols - 1 : bottomLeftCol + 1;
			float bottomLeftValue= ratio * sum.at<int>(addOneRow, addOneCol) + (1 - ratio)*sum.at<int>(bottomLeftRow, bottomLeftCol);
			//右上角带小数的sum值
			float topRightRow = int(wRect.r.y);
			float topRightCol = int(wRect.r.x + wRect.r.width);
			ratio = (wRect.r.y - topRightRow)*(wRect.r.x + wRect.r.width- topRightCol);
			addOneRow = topRightRow + 1 > sum.rows - 1 ? sum.rows - 1 : topRightRow + 1;
			addOneCol = topRightCol + 1 > sum.cols - 1 ? sum.cols - 1 : topRightCol + 1;
			float topRightValue = ratio * sum.at<int>(addOneRow, addOneCol) + (1 - ratio)*sum.at<int>(topRightRow, topRightCol);
			featureValue += wRect.weight*(topLeftValue
				+ bottomRightValue
				- bottomLeftValue
				- topRightValue);
		}
	}
	//近似的计算delta值
	cv::Rect nearRect(detect.x,detect.y,detect.width,detect.height);
	float delta=FeatureManager::calcDelta(sum, sumSq, nearRect);
	return featureValue/delta;
}

void Feature::save(cv::FileStorage & fs)
{
	for (int i = 0; i < MAX_RECT_COUNT; i++)
	{
		fs << "weight" + to_string(i) << rect[i].weight;
		fs << "r" + to_string(i) << rect[i].r;
	}
}

void Feature::load(cv::FileNode & node)
{
	for (int i = 0; i < MAX_RECT_COUNT; i++)
	{
		node["weight" + to_string(i)] >> rect[i].weight;
		node["r" + to_string(i)]>> rect[i].r;
	}
}

Feature operator*(float x, const Feature & y)
{
	return y*x;
}

Feature operator*(const Feature & y, float x)
{
	Feature::WeightRect newRect[MAX_RECT_COUNT];
	for (int i = 0; i < MAX_RECT_COUNT; i++)
	{
		newRect[i].weight = y.rect[i].weight;
		newRect[i].r.x = y.rect[i].r.x*x;
		newRect[i].r.y = y.rect[i].r.y*x;
		newRect[i].r.width = y.rect[i].r.width*x;
		newRect[i].r.height =y.rect[i].r.height*x;
	}
	return Feature(newRect[0].weight, newRect[0].r, newRect[1].weight, newRect[1].r, newRect[2].weight, newRect[2].r);
}
