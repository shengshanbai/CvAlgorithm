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
    char* equalized_window = "Equalized Image";
    /// Load image
    src = imread( argv[1], 1 );
    if(!src.data || argc<3) {
        cout<< "Usage:Hequlization <path to image>" <<endl;
        return -1;
    }
    int mode=atoi(argv[2]);
    cvtColor(src,src,COLOR_BGR2GRAY);
    switch (mode) {
    case 1:
        /// Apply Histogram Equalization
        equalizeHist( src, dst );
        break;
    case 2:
        ///Apply CLAHE
    {
        auto pAHE=createCLAHE();
        pAHE->apply(src,dst);
        break;
    }
    default:
        break;
    }
    /// Display results
    namedWindow( source_window, WINDOW_AUTOSIZE);
    namedWindow( equalized_window, WINDOW_AUTOSIZE);
    imshow(source_window,src);
    imshow(equalized_window,dst);
    waitKey(0);
    return 0;
}
