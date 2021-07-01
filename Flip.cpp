#include <iostream>
#include <time.h>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

void cvFlip();
void cvRotation();
void cvAffine();
void cvPerspective();
void getmyPerspective();
Mat getmyRotationMatrix(Point center, double x, double y);
void myRotation();
Mat myTransMat();
void myPerspective();
void cvbasedmyHarrisCorner();
void cvFeaturesSIFT();

void main()
{
	/*cvFlip();
	cvRotation();
	cvAffine();
	getmyPerspective();
	cvPerspective();*/
	//1번
	cvRotation();
	myRotation();

	//myPerspective();
	cvbasedmyHarrisCorner();
	//cvFeaturesSIFT();
}

void cvFlip()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst_x, dst_y, dst_xy;

	flip(src, dst_x, 0);
	flip(src, dst_y, 1);
	flip(src, dst_xy, -1);

	imwrite("nonflip.jpg", src);
	imwrite("xflip.jpg", dst_x);
	imwrite("yflip.jpg", dst_y);
	imwrite("xyflip.jpg", dst_xy);

	imshow("nonflip", src);
	imshow("xflip", dst_x);
	imshow("yflip", dst_y);
	imshow("xyflip", dst_xy);
	waitKey(0);

	destroyAllWindows();
}

void cvRotation()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);
	matrix = getRotationMatrix2D(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());

	imwrite("nonrot.jpg", src);
	imwrite("rot.jpg", dst);
	
	imshow("nonrot", src);
	imshow("rot", dst);
	waitKey(0);

	destroyAllWindows();
}
void myRotation()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);
	matrix = getmyRotationMatrix(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());

	imwrite("nonrot.jpg", src);
	imwrite("rot.jpg", dst);

	imshow("mynonrot", src);
	imshow("myrot", dst);
	waitKey(0);

	destroyAllWindows();
}
Mat getmyRotationMatrix(Point center, double x, double y)
{
	const float PI = 3.141592;
	float mcos = cos(x * PI / 180);
	float a = y * mcos;
	float msin = sin(x * PI / 180);
	float b = y * msin;
	/*float data[] ={a , b, (1 - a) * center.x - b * center.y, 
					-b, a, b * (center.x) + (1 - a) * center.y};*/
	//Mat dst = Mat::zeros(2,3,CV_32FC3);
	Mat dst = (Mat_<float>(2, 3) << a, b, (1 - a) * center.x - b * center.y,
		-b, a, b * (center.x) + (1 - a) * center.y); // 매트릭스를 직접 만듦

	return dst;


}
void cvAffine()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point2f srcTri[3];
	srcTri[0] = Point2f(0.f, 0.f);
	srcTri[1] = Point2f(src.cols - 1.f, 0.f);
	srcTri[2] = Point2f(0.f, src.rows - 1.f);

	Point2f dstTri[3];
	dstTri[0] = Point2f(0.f, src.rows * 0.33f);
	dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);

	matrix = getAffineTransform(srcTri, dstTri);
	warpAffine(src, dst, matrix, src.size());

	imwrite("nonaff.jpg", src);
	imwrite("aff.jpg", dst);

	imshow("nonaff", src);
	imshow("aff", dst);
	waitKey(0);

	destroyAllWindows();

}

void getmyPerspective()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point2f srcQuad[4];
	srcQuad[0] = Point2f(0.f, 0.f);
	srcQuad[1] = Point2f(src.cols - 1.f, 0.f);
	srcQuad[2] = Point2f(0.f, src.rows - 1.f);
	srcQuad[3] = Point2f(src.cols -1.f, src.rows - 1.f);

	Point2f  dstQuad[4];
	dstQuad[0] = Point2f(0.f, src.rows * 0.33f);
	dstQuad[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstQuad[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
	dstQuad[3] = Point2f(src.cols * 0.85f, src.rows * 0.7f);

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	imwrite("nonper.jpg", src);
	imwrite("per.jpg", dst);

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);

	destroyAllWindows();
}
void myPerspective()
{
	Mat src = imread("card_per.png", 1);
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY); // gray scale로 변환
	Mat dst, matrix;
	Mat harr;
	cornerHarris(gray, harr, 4, 5, 0.04, BORDER_DEFAULT); // corner detect
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	vector<float> rowvec;
	vector<float> colvec;
	int thresh = 100;
	Mat result = src.clone();
	for (int y = 0; y < harr.rows; y += 1)
	{
		for (int x = 0; x < harr.cols; x += 1)
		{
			if ((int)harr.at<float>(y, x) > thresh)
			{
				rowvec.push_back(x);
				colvec.push_back(y);
			}
		}
	}
	float min = *min_element(rowvec.begin(), rowvec.end()); //최솟값 찾기


	Point2f srcQuad[4];
	srcQuad[0] = Point2f(0.f, 0.f);
	srcQuad[1] = Point2f(src.cols - 1.f, 0.f);
	srcQuad[2] = Point2f(0.f, src.rows - 1.f);
	srcQuad[3] = Point2f(src.cols - 1.f, src.rows - 1.f);

	Point2f  dstQuad[4];
	dstQuad[0] = Point2f(0.f, src.rows * 0.33f);
	dstQuad[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstQuad[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
	dstQuad[3] = Point2f(src.cols * 0.85f, src.rows * 0.7f);

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	imwrite("card_nonper.jpg", src);
	imwrite("card_per.jpg", dst);

	imshow("card_nonper", src);
	imshow("card_per", dst);
	waitKey(0);

	destroyAllWindows();
}
void cvbasedmyHarrisCorner()
{
	Mat img = imread("card_nonper.jpg",1);
	if (img.empty())
	{
		cout << "Empty image!\n";
		exit(-1);
	}

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	//<Do Harris corner detection >

	Mat harr;
	cornerHarris(gray, harr, 4, 5, 0.04, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//<Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);
	vector<float> rowvec;
	vector<float> colvec;
	//<Print corners>
	int thresh = 100;
	Mat result (img.rows, img.cols, CV_8UC1);
	Mat result2 = img.clone();
	for (int y = 0; y < harr.rows; y += 1)
	{
		for (int x = 0; x < harr.cols; x += 1)
		{
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result, Point(x, y), 15, Scalar(0, 0, 0), -1);
				//rowvec.push_back(y);
				//colvec.push_back(x);
			}
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result2, Point(x, y), 15, Scalar(0, 0, 0), -1);
				//rowvec.push_back(y);
				//colvec.push_back(x);
			}
		}
	}
	imwrite("targeting.png", result);
	Mat targeting = imread("targeting.png");
	// <Set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 0;
	params.maxThreshold = 3000;
	params.filterByArea = true;
	params.minArea = 16;
	params.maxArea = 10000;
	params.filterByCircularity = true;
	params.minCircularity = 0.300;
	params.filterByConvexity = true;
	params.minConvexity = 0.5;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.0001;
	/*for (int sz=0; sz < rowvec.size(); sz++)
	{
		cout << rowvec[sz] << " ";
	}*/
	// <Set blob detector > 
	Ptr < SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs > 
	std::vector<KeyPoint>keypoints;
	detector->detect(targeting, keypoints);
	for (int sz=0; sz < rowvec.size(); sz++)
	{
		cout << keypoints[sz].pt << " ";
	}
	//<Draw blobs>
	Mat resultblob;
	drawKeypoints(targeting, keypoints, resultblob, Scalar(0, 0, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("Source image", img);
	//imshow("Harris image", harr_abs);
	//Mat concat;
	imshow("result corner", result2);
	//hconcat(result, resultblob , concat);
	imshow("Target image", result);
	imshow("target_blob", resultblob);
	//imshow("result", concat);
	Point2f srcQuad[4];
	srcQuad[0] = Point2f(keypoints[0].pt);
	srcQuad[1] = Point2f(keypoints[1].pt);
	srcQuad[2] = Point2f(keypoints[2].pt);
	srcQuad[3] = Point2f(keypoints[3].pt);

	Point2f  dstQuad[4];
	/*dstQuad[0] = Point2f(0 , 50);
	dstQuad[1] = Point2f(img.cols, 50);
	dstQuad[2] = Point2f(img.cols, 200);
	dstQuad[3] = Point2f(0 , 200);*/
	dstQuad[0] = Point2f(50, 350);
	dstQuad[1] = Point2f(img.cols-50, 350);
	dstQuad[2] = Point2f(img.cols-50, 100);
	dstQuad[3] = Point2f(50, 100);
	Mat matrix;
	Mat dst;
	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(img, dst, matrix, img.size());

	/*imwrite("card_nonper.jpg", src);
	imwrite("card_per.jpg", dst);*/

	imshow("card_nonper", img);
	imshow("card_per", dst);
	waitKey(0);

	destroyAllWindows();

}
//void cvFeaturesSIFT()
//{
//	Mat img = cv::imread("card_nonper.jpg", 1);
//	Mat reimg;
//	resize(img, reimg, Size(512, 512));
//	Mat gray;
//	cvtColor(reimg, gray, CV_BGR2GRAY);
//
//	Ptr <cv::SiftFeatureDetector> detector = SiftFeatureDetector::create();
//	std::vector<KeyPoint> keypoints;
//	detector->detect(gray, keypoints);
//
//	Mat result;
//	drawKeypoints(reimg, keypoints, result);
//	imwrite("sift_result.jpg", result);
//	imshow("Sift result", result);
//	waitKey(0);
//	destroyAllWindows();
//}
void cvPerspective()
{
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	matrix = myTransMat();
	warpPerspective(src, dst, matrix, src.size());

	imwrite("nonper.jpg", src);
	imwrite("per.jpg", dst);

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);

	destroyAllWindows();
}

Mat myTransMat()
{
	Mat matrix1 = (Mat_<double>(3, 3) << 1, tan(45 * CV_PI / 180), 0,
		0, 1, 0,
		0, 0, 1);
	Mat matrix2 = (Mat_<double>(3, 3) << 1, 0, -256,
		0, 1, 0,
		0, 0, 1);

	Mat matrix3 = (Mat_<double>(3, 3) << 0.5, 0, 0,
		0, 0.5, 0,
		0, 0, 1);

	return matrix3 * matrix2 * matrix1;
}