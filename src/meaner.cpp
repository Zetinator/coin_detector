#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

vector<Mat3b> monedas;
Mat3b moneda;

Mat3b getMean(const vector<Mat3b>& images)
{
  if (images.empty()) return Mat3b();

    // Create a 0 initialized image to use as accumulator
  Mat m(images[0].rows, images[0].cols, CV_64FC3);
  m.setTo(Scalar(0,0,0,0));

    // Use a temp image to hold the conversion of each input image to CV_64FC3
    // This will be allocated just the first time, since all your images have
    // the same size.
  Mat temp;
  for (int i = 0; i < images.size(); ++i)
  {
        // Convert the input images to CV_64FC3 ...
    images[i].convertTo(temp, CV_64FC3);

        // ... so you can accumulate
    m += temp;
  }

    // Convert back to CV_8UC3 type, applying the division to get the actual mean
  m.convertTo(m, CV_8U, 1. / images.size());
  return m;
}

int main()
{
  vector<Mat3b> images;
  String output_folder1 = "./detectedCoins";
  for (int i = 1; i < 22; ++i)
  {
    moneda = imread(format("%s/c%d.png", output_folder1.c_str(),i),1);
    images.push_back(moneda);
  }

    // Compute the mean
  Mat3b meanImage = getMean(images);
  String output_folder = ".";
  imwrite(format("%s/mean20%d.png", output_folder.c_str(),1), meanImage);

    // Show result
  imshow("Mean image", meanImage);
  waitKey();

  return 0;
}