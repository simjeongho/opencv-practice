#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace std;
using namespace cv;

void cvHarrisCorner();
void myHarrisCorner();
void myBlobDetection();
void cvBlobDetection();
void cvFeaturesSIFT();
void mycornerblobDetection(Mat img);
void warpPers();
void oriwarpPers();
void mycvFeaturesSIFT();

void main()
{
	Mat triangle = imread("tri2.png");
	Mat rect = imread("rect1.png");
	Mat penta = imread("penta1.png");
	Mat hexa = imread("hexa1.png");
	
	//1번

	cvBlobDetection();
	//waitKey(0);

	//2번
	mycornerblobDetection(triangle);
	mycornerblobDetection(rect);
	mycornerblobDetection(penta);
	mycornerblobDetection(hexa);
	

	//3번
	cvFeaturesSIFT();
	mycvFeaturesSIFT();
}

void cvHarrisCorner()
{
	Mat img = imread("ship.png");
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
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//<Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	//<Print corners>
	int thresh = 125;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1)
	{
		for (int x = 0; x < harr.cols; x += 1)
		{
			if ((int)harr.at<float>(y, x) > thresh)
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
		}
	}

	imshow("Source image", img);
	imshow("Harris image", harr_abs);
	imshow("Target image", result);
	//waitKey(0);
	//destroyAllWindows();
}

void myHarrisCorner()
{
	Mat img = imread("ship.png");
	if (img.empty())
	{
		cout << "Empty mage!\n";
		exit(-1);
	}

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	int height = gray.rows;
	int width = gray.cols;

	//<Get gradient>
	Mat blur;
	GaussianBlur(gray, blur, Size(3, 3), 1);

	Mat gx, gy;
	cv::Sobel(blur, gx, CV_64FC1, 1, 0, 3, 0.4, 128);
	cv::Sobel(blur, gy, CV_64FC1, 0, 1, 3, 0.4, 128);
	double* gx_data = (double*)(gx.data);
	double* gy_data = (double*)(gy.data);

	//<Get score>
	Mat harr = Mat(height, width, CV_64FC1, Scalar(0));
	double* harr_data = (double*)(harr.data);
	double k = 0.02;

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			int center = y * width + x;

			double dx = 0, dy = 0, dxdy = 0;
			for (int u = -1; u <= 1; u++)
			{
				for (int v = -1; v <= 1; v++)
				{
					int cur = center + u * width + v;

					double ix = *(gx_data + cur);
					double iy = *(gy_data + cur);
					dx += ix * ix;
					dy += iy * iy;
					dxdy += ix * iy;
				}
			}
			*(harr_data + center)
				= dx * dy - dxdy * dxdy - k * (dx + dy) * (dx + dy);
		}
	}
	//<Detect corner by score>
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			int center = y * width + x;
			double value = *(harr_data + center);

			bool isMaximum = true, isMinimum = true;
			for (int u = -1; u <= 1; u++)
			{
				for (int v = -1; v <= 1; v++)
				{
					if (u != 0 || v != 0)
					{
						int cur = center + u * width + v;

						double neighbor = *(harr_data + cur);
						if (value < neighbor) isMaximum = false;
						else if (value > neighbor) isMinimum = false;
					}
				}
			}
			if (isMaximum == false && isMinimum == false)
				*(harr_data + center) = 0;
			else
				*(harr_data + center) = value;
		}
	}

	 // <Print corners>
	Mat result = img.clone();
	double thresh = 0.1;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			int center = y * width + x;

			if (*(harr_data + center) > thresh)
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
		}
	}
	imshow("Source image", img);
	imshow("Target image", result);
	//waitKey(0);
	//destroyAllWindows();
}

void mycornerblobDetection(Mat img)
{
	//Mat img = imread("tri2.png");
	if (img.empty())
	{
		cout << "Empty image!\n";
		exit(-1);
	}
	//resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	//<Do Harris corner detection >

	Mat harr;
	cornerHarris(gray, harr, 5, 3, 0.06, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//<Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	//<Print corners>
	int thresh = 125;
	Mat result(img.rows, img.cols, CV_8UC1);
	Mat result2 = img.clone();
	for (int y = 0; y < harr.rows; y += 1)
	{
		for (int x = 0; x < harr.cols; x += 1)
		{
			if ((int)harr.at<float>(y, x) > thresh)
				circle(result, Point(x, y), 25, Scalar(0, 0, 0), -1);
			if ((int)harr.at<float>(y, x) > thresh)
				circle(result2, Point(x, y), 25, Scalar(0, 0, 0), -1);
		}

	}
	imwrite("Target_blob.png", result);
	imshow("Target image", result2);
	//imshow("src img", img);
	Mat target = imread("Target_blob.png");
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

	// <Set blob detector > 
	Ptr < SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs > 
	std::vector<KeyPoint>keypoints;
	detector->detect(target, keypoints);

	//<Draw blobs>
	Mat resultblob;
	drawKeypoints(target, keypoints, resultblob, Scalar(127, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cout << "이 도형은 " << keypoints.size() << "각형 입니다." <<   endl;
	imshow("blob result", resultblob);
	waitKey(0);
	//destroyAllWindows();
}


void myBlobDetection()
{
	Mat src, src_gray, dst;
	src = imread("butt.jpg", 1);
	cvtColor(src, src_gray, COLOR_RGB2GRAY);

	int gau_ksize = 11;
	int iap_ksize = 3;
	int iap_scale = 1;
	int iap_delta = 1;

	GaussianBlur(src_gray, src_gray, Size(gau_ksize, gau_ksize), 3, 3, BORDER_DEFAULT);
	Laplacian(src_gray, dst, CV_64F, iap_ksize, iap_scale, iap_delta, BORDER_DEFAULT);
	 // Gaussizn + Laplacian -> LOG

	normalize(-dst, dst, 0, 255, NORM_MINMAX, CV_8U, Mat());

	imwrite("my_log_dst.png", dst);
	imshow("Original image", src);
	imshow("Laplacian Image", dst);
	waitKey(0);
	destroyAllWindows();
}

void cvBlobDetection()
{
	Mat img = imread("coin.png", IMREAD_COLOR);
	
	 // <Set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 0;
	params.maxThreshold = 10000;
	params.filterByArea = true;
	params.minArea = 16;
	params.maxArea = 1000000;
	params.filterByCircularity = true;
	params.minCircularity = 0.800;
	params.filterByConvexity = true;
	params.minConvexity = 0.6;
	//params.filterByInertia = true;
	params.minInertiaRatio = 0.0001;

	// <Set blob detector > 
	Ptr < SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs > 
	std::vector<KeyPoint>keypoints;
	detector->detect(img, keypoints);

	//<Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cout <<  "동전의 개수는 " << keypoints.size() << endl;
	imshow("keypoints", result);
	waitKey(0);
	destroyAllWindows();
}

void cvFeaturesSIFT()
{
	Mat img = cv::imread("church.jpg", 1);
	Mat reimg;
	resize(img, reimg, Size(512, 512));
	Mat gray;
	cvtColor(reimg, gray, CV_BGR2GRAY);

	Ptr <cv::SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);

	Mat result;
	drawKeypoints(reimg, keypoints, result);
	imwrite("sift_result.jpg", result);
	imshow("Sift result", result);
	waitKey(0);
	destroyAllWindows();
}
void mycvFeaturesSIFT()
{
	Mat src = imread("church.jpg", 1);
	Mat dst;
	Mat resrc;
	resize(src, resrc, Size(512, 512));
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(512, 0);
	src_p[2] = Point2f(0, 512);
	src_p[3] = Point2f(512, 512);


	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(512, 0);
	dst_p[2] = Point2f(0, 512);
	dst_p[3] = Point2f(412, 412);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(resrc, dst, pers_mat, Size(512, 512));
	Mat result = dst * 0.7;

	Mat gray;
	cvtColor(result, gray, CV_BGR2GRAY);

	Ptr <cv::SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);

	Mat resultSIFT;
	drawKeypoints(result, keypoints, resultSIFT);
	imwrite("sift_result2.jpg", resultSIFT);
	imshow("Sift result", resultSIFT);
	waitKey(0);
	destroyAllWindows();
}
void warpPers()
{
	Mat src = imread("church.jpg", 0);
	Mat dst;
	Mat resrc;
	resize(src, resrc, Size(512, 512));
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(512, 0);
	src_p[2] = Point2f(0, 512);
	src_p[3] = Point2f(512, 512);


	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(512, 0);
	dst_p[2] = Point2f(0, 512);
	dst_p[3] = Point2f(412, 412);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(resrc, dst, pers_mat, Size(512, 512));
	Mat result = dst * 1.5;

	imshow("dst", dst);
	imshow("src", src);
	imshow("result", result);
	imshow("resrc", resrc);
	waitKey(0);

}
void oriwarpPers()
{
	Mat src = imread("church.jpg", 0);
	Mat dst;
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(512, 0);
	src_p[2] = Point2f(0, 512);
	src_p[3] = Point2f(512, 512);


	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(512, 0);
	dst_p[2] = Point2f(0, 512);
	dst_p[3] = Point2f(512, 512);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(src, dst, pers_mat, Size(512, 512));
	Mat result = dst * 1.5;

	imshow("dst", dst);
	imshow("src", src);
	imshow("result", result);
	waitKey(0);

}
