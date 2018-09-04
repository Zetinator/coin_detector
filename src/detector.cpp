#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>


using namespace cv;
using namespace std;

Mat src, src_gray;
Mat dst, detected_edges;
Mat enhanced;

Mat src_base, hsv_base;
Mat src_test1, hsv_test1;

// <<<<<<Canny 45 Circle 50<<<<<<
int lowCannyThreshold = 45;
int lowCircleThreshold = 50;
int const max_lowCannyThreshold = 100;
int const max_lowCircleThreshold = 200;
int ratio = 3;
int kernel_size = 3;
int x;
int y;
int width;
int height;
int couter10;
int counter50c;
int counter5;
int counter1;

double maxR;
double minR;
double ratio10_5 = 1.098039216;
double ratio10_1 = 1.333333333;
double ratio10_50c = 1.647058824;

double ratio50c_10 = 0.608369099;
double ratio50c_5 = 0.663157895;
double ratio50c_1 = 0.808844508;

double epsilon = 0.275442;

String text;
int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
double fontScale = 1;
int thickness = 3;


String window_name = "Target";


int normalize(int valor)
{
	if (valor <= 0)
	{
		valor = 1;
	}
	return(valor);
}

double maxRadius(vector<Vec3f> circles) {
	double max = 0;
	for (size_t i = 0; i < circles.size(); ++i) {
		if (max < circles[i][2]) {
			max = circles[i][2];
		}
	}
	return max;
}
double minRadius(vector<Vec3f> circles) {
	double min = 99999;
	for (size_t i = 0; i < circles.size(); ++i) {
		if (min > circles[i][2]) {
			min = circles[i][2];
		}
	}
	return min;
}

double find10(Mat imageToAnalize)
{
	src_base = imread( "/home/erickzetinator/Dokumente/PDS/src/proyecto_final/reference/mean101.png", 1 );
	src_test1 = imageToAnalize;
	/// Convert to HSV
	cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
	cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };
	/// Histograms
	MatND hist_base;
	MatND hist_test1;

	/// Calculate the histograms for the HSV images
	calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
	normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
	calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
	normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Apply the histogram comparison methods
	int compare_method = 0; 
	double base_test1 = compareHist( hist_base, hist_test1, compare_method );

	return(base_test1);
}

double find50c(Mat imageToAnalize)
{
	src_base = imread( "/home/erickzetinator/Dokumente/PDS/src/proyecto_final/reference/mean201.png", 1 );
	src_test1 = imageToAnalize;
	/// Convert to HSV
	cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
	cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };
	/// Histograms
	MatND hist_base;
	MatND hist_test1;

	/// Calculate the histograms for the HSV images
	calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
	normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
	calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
	normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Apply the histogram comparison methods
	int compare_method = 0; 
	double base_test1 = compareHist( hist_base, hist_test1, compare_method );

	return(base_test1);
}

void CannyThreshold(int, void*)
{
	blur( src_gray, detected_edges, Size(3,3) );
	lowCannyThreshold = normalize(lowCannyThreshold);
	Canny( detected_edges, detected_edges, lowCannyThreshold, lowCannyThreshold*ratio, kernel_size );

	dst = Scalar::all(0);
	src.copyTo( dst, detected_edges);

	Size sizeA(dst.cols/1.8,dst.rows/1.8);
	resize(dst,dst,sizeA);
	imshow( window_name, dst );
}


void CircleThreshold(int, void*)
{
	couter10 = 0;
	counter50c = 0;
	counter5 = 0;
	counter1 = 0;

	GaussianBlur( src_gray, detected_edges, Size(9,9), 0,0);
	Canny(detected_edges, detected_edges, lowCannyThreshold, lowCannyThreshold*ratio, kernel_size );

	vector<Vec3f> circles;
	lowCircleThreshold = normalize(lowCircleThreshold);
	HoughCircles( detected_edges, circles, CV_HOUGH_GRADIENT, 2, src_gray.rows/20, 10, lowCircleThreshold, 20, 60);

	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);

	maxR = maxRadius(circles);
	cout<<"MaxRadius : "<<maxR <<endl;

	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		Mat coin = Mat(enhanced, cv::Rect(circles[i][0] - radius,circles[i][1]-radius/5,radius*2,radius/5)).clone();
		ellipse(dst, center, Size(radius,radius), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0 );

		// string output_folder = "./detectedCoins";
		// Size sizeS(200,20);
		// resize(coin,coin,sizeS);
		// imwrite(format("%s/detectedCoin%d.png", output_folder.c_str(),i), coin);

		cout<<"correlation10: "<<find10(coin) <<endl;
		cout<<"correlation50c: "<<find50c(coin) <<endl;
		cout<<"ratio: "<<maxR/radius <<endl;



		if (find10(coin) >= .35 || maxR/radius < 1.04)
		{
			couter10 = couter10 + 1;
			cout << "detected: 10" <<endl;
			putText(dst, "10", center, fontFace, fontScale,
					Scalar::all(255), thickness, 8);
		}else{
			if (find50c(coin) >= .1 || maxR/radius >= 1.6)
				// if (find50c(coin) >= .1 || abs(maxR/radius - ratio10_50c) <= epsilon)
			{
				counter50c = counter50c + 1;
				cout << "detected: 50c" <<endl;
				putText(dst, "50c", center, fontFace, fontScale,
						Scalar::all(255), thickness, 8);
			}else{
				if (maxR/radius >= 1.04 && maxR/radius <= 1.2) 
					// if (abs(maxR/radius - ratio10_5) <= epsilon)
				{
					counter5 = counter5 + 1;
					cout << "detected: 5" <<endl;
					putText(dst, "5", center, fontFace, fontScale,
							Scalar::all(255), thickness, 8);
				}else{
					if (maxR/radius >= 1.3 && maxR/radius <= 1.5)
					{
						counter1 = counter1 + 1;
						cout << "detected: 1" <<endl;
						putText(dst, "1", center, fontFace, fontScale,
								Scalar::all(255), thickness, 8);
					}else{
						cout << "detected: ?" <<endl;
						putText(dst, "?", center, fontFace, fontScale,
								Scalar::all(255), thickness, 8);
					}
				}
			}
		}
	}

	Size sizeD(dst.cols/1.8,dst.rows/1.8);
	resize(dst,dst,sizeD);
	imshow( window_name, dst );
	cout<<"10: " <<couter10 <<"    -50c: " <<counter50c <<"    5: " <<counter5 <<"    1: " << counter1 <<endl; 
}



int main( int argc, char** argv )
{
	/// Load
	src = imread( argv[1] );
	if( !src.data )
	{return -1;}

	GaussianBlur( src, src, Size(3,3), 0,0);

	// contraste
	Mat new_image = Mat::zeros( src.size(), src.type() );
	for( int y = 0; y < src.rows; y++ )
	{ for( int x = 0; x < src.cols; x++ )
		{ for( int c = 0; c < 3; c++ )
			{
				new_image.at<Vec3b>(y,x)[c] =
					saturate_cast<uchar>( 1.5*( src.at<Vec3b>(y,x)[c] ));
			}
		}
	}

	int erosion_size = 3;
	Mat element = getStructuringElement( MORPH_ELLIPSE,
			Size( 2*erosion_size + 1, 2*erosion_size+1 ),
			Point( erosion_size, erosion_size ) );

	/// Apply the erosion operation

	erode( new_image, new_image, element );

	int dilation_size = 3;
	element = getStructuringElement( MORPH_ELLIPSE,
			Size( 2*dilation_size + 1, 2*dilation_size+1 ),
			Point( dilation_size, dilation_size ) );
	/// Apply the dilation operation

	dilate( new_image, new_image, element );
	enhanced = new_image;
	dst.create( src.size(), src.type() );
	cvtColor( new_image, src_gray, CV_BGR2GRAY );

	Size sizeL(new_image.cols/1.8,new_image.rows/1.8);
	resize(new_image,new_image,sizeL);
	imshow("Preview",new_image);
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	/// Trackbar
	createTrackbar( "ThresholdCircle:", window_name, &lowCircleThreshold, max_lowCircleThreshold, CircleThreshold );
	createTrackbar( "ThresholdCanny:", window_name, &lowCannyThreshold, max_lowCannyThreshold, CannyThreshold );

	/// Create a Trackbar for user to enter threshold
	/// ShowTime
	CircleThreshold(0, 0);

	printf( "Done \n" );

	waitKey(0);
	return 0;
}
