#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<cstdio>
#include<cstdlib>

using namespace cv;
using namespace std;

int main(int argc,char** argv) {
    Mat src,dst;
    char* source_window = "Source image";
    char* filtered_window = "Filtered Image";
    /// Load image
    src = imread( argv[1], 1 );
    if(!src.data || argc<3) {
        cout<< "Usage:Lfilter <path to image> <mode>" <<endl;
        return -1;
    }
    int mode=atoi(argv[2]);
    switch (mode) {
    case 1:
    {
        //默认的盒式滤波
        boxFilter(src,dst,-1,Size(3,3));
    }
        break;
    case 2:
    {
        //锐化图片
        float kernel[][3]={-1,-1,-1,
                          -1,9,-1,
                          -1,-1,-1};
        Mat kerMat(3,3,CV_32F,kernel);
        filter2D(src,dst,-1,kerMat);
    }
        break;
    default:
        break;
    }
    /// Display results
    namedWindow( source_window, WINDOW_AUTOSIZE);
    namedWindow( filtered_window, WINDOW_AUTOSIZE);
    imshow(source_window,src);
    imshow(filtered_window,dst);
    waitKey(0);
    return 0;
}
