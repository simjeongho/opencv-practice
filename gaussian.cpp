#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;
using namespace std;
int myKernelCon9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height); //convolution
int myKernelCon3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);
Mat myGaussianfilter(Mat srcImg);
Mat myCopy(Mat srcImg);
Mat GetHistogram(Mat& src);
void SpreadSalts_blue(Mat img, int num);
Mat mySobelFilter(Mat srcImg);
vector <Mat> myGaussianPyramid(Mat srcImg);
Mat mySampling(Mat srcImg);
Mat myGaussianfiltercolor(Mat srcImg);
int myKernelCon3x3color(Vec3b* arr, int kernel[][3], int x, int y, int width, int height); //convolution
int main()
{
	Mat gear = imread("gear.jpg", 0); // ���� �̹���
	Mat colorgear = imread("gear.jpg", 1); // �÷�
	imshow("gear", gear); // ���� �̹���

	Mat bgear = GetHistogram(gear); // ����þ� �� ������׷�
	imshow("before gaussian", bgear); // ����þ� �� ������׷�

	Mat blurgear = myGaussianfilter(gear); // ����þ� ó��

	imshow("gear1", blurgear); // ����þ� �� �̹���
	Mat agear = GetHistogram(blurgear); // ����þ� �� �̹���
	imshow("after gaussian", agear); // ����þ� �� �̹��� �����ֱ�

	SpreadSalts_blue(gear, 100); // �� ���
	imshow("salt gear", gear); // ������ �̹��� �����ֱ�
	Mat saltblurgear = myGaussianfilter(gear); // �� ���� �� �� ó���ϱ� (����þ�)
	imshow("salt after gaussian", saltblurgear); // �����ֱ�

	Mat sobelgear = mySobelFilter(gear); // 45 135�� �Һ� ����
	imshow("sobel", sobelgear); // ó�� ��

	vector <Mat> gau = myGaussianPyramid(gear);
	imshow("2nd", gau[2]);
	imshow("1st", gau[1]);
	imshow("0st", gau[0]);
	imshow("colorgear", colorgear);


	waitKey(0);
}

Mat myCopy(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dstData[y * width + x] = srcData[y * width + x];
		}
	}
	return dstImg;
}

int myKernelCon9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height) //convolution
{
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 7; j++)
	{
		for (int i = -1; i <= 7; i++)
		{
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
			{
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) { return sum / sumKernel; } //���� 1�� ����ȭ�ǵ��� �� ������ ��� ��ȭ ����
	else { return sum; }
}
int myKernelCon3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) //convolution
{
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
			{
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) { return sum / sumKernel; } //���� 1�� ����ȭ�ǵ��� �� ������ ��� ��ȭ ����
	else { return sum; }
}

Mat myGaussianfilter(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;

	int kernel[9][9] = { 0,1,1,2,2,2,1,1,0,
						 1,2,4,5,5,5,4,2,1,
						 1,4,5,3,0,3,5,4,1,
						 2,5,3,12,24,12,3,5,2,
						 2,5,0,24,40,24,0,5,2,
						 2,5,3,12,24,12,3,5,2,
						 1,4,5,3,0,3,5,4,1,
						 1,2,4,5,5,5,4,2,1,
						 0,1,1,2,2,2,1,1,0
};
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dstData[y * width + x] = myKernelCon9x9(srcData, kernel, x, y, width, height);
		}
	}

	return dstImg;
}
int myKernelCon3x3color(Vec3b* arr, int kernel[][3], int x, int y, int width, int height) //convolution
{
	//int sum = 0;
	int chRed, chGreen, chBlue;
	int fvalRed, fvalGreen, fvalBlue;
	float sumkernelred, sumkernelgreen, sumkernelblue;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
			{
				
				//sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	//if (sumKernel != 0) { return sum / sumKernel; } //���� 1�� ����ȭ�ǵ��� �� ������ ��� ��ȭ ����
	//else { return sum; }
}

//Mat myGaussianfiltercolor(Mat srcImg)
//{
//	int width = srcImg.cols;
//	int height = srcImg.rows;
//
//	int kernel[3][3] = {1,2,1,
//						2,4,2,
//						1,2,1};
//	Mat dstImg(srcImg.size(), CV_8UC3);
//	/*Vec3b* srcData = srcImg.data;
//	Vec3b* dstData = dstImg.data;*/
//	for (int y = 0; y < height; y++)
//	{
//		for (int x = 0; x < width; x++)
//		{
//			Vec3b* srcData = srcImg.at<Vec3b>(y, x)[0];
//		}
//	}
//
//	for (int y = 0; y < height; y++)
//	{
//		for (int x = 0; x < width; x++)
//		{
//			dstImg.at<Vec3b>(y, x)[0] = srcImg.at<Vec3b>(y * 2, x * 2)[0];
//			dstImg.at<Vec3b>(y, x)[1] = srcImg.at<Vec3b>(y * 2, x * 2)[1];
//			dstImg.at<Vec3b>(y, x)[2] = srcImg.at<Vec3b>(y * 2, x * 2)[2];
//			dstData[y * width + x] = myKernelCon3x3color(srcData, kernel, x, y, width, height);
//		}
//	}
//
//	return dstImg;
//}

Mat mySamplingcolor(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstImg.at<Vec3b>(y, x)[0] = srcImg.at<Vec3b>(y * 2, x * 2)[0];
			dstImg.at<Vec3b>(y, x)[1] = srcImg.at<Vec3b>(y * 2, x * 2)[1];
			dstImg.at<Vec3b>(y, x)[2] = srcImg.at<Vec3b>(y * 2, x * 2)[2];
		}
	}
	return dstImg;
}

Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//������׷� ���
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//������׷� plot
	int hist_w = 500;
	int hist_h = 1000;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());//����ȭ 

	for (int i = 1; i < number_bins; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0); // ���� ���� �մ� ���� �׸��� ������� plot
	}
	return histImage;
}
void SpreadSalts_blue(Mat img, int num)

{
	//num: ���� ���� ����
	for (int n = 0; n < num; n++)
	{
		int x = rand() % img.cols;//img.cols�� �̹����� �� ������ ����
		int y = rand() % img.rows; // img.rows�� �̹����� ���� ������ ����
		//�������� ������ ���� ���� �� �����Ƿ� ������ ��ġ�� 
		//�̹����� ũ�⸦ ����� �ʵ��� �����ϴ� ������ �Ͽ���

		if (img.channels() == 1)
		{
			//img.channels()�� �̹����� ä�� ���� ��ȯ
			img.at<uchar>(y, x) = 255;

		}
		else
		{
			img.at<Vec3b>(y, x)[0] = 255; // Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 0; // Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0; // Red ä�� ����
		}
	}
}

Mat mySobelFilter(Mat srcImg)
{
	int kernelX[3][3] = { -1, -1,2,
						-1,2,-1,
						2 ,-1,-1 }; // 45�� ����ũ
	int kernelY[3][3] = { 2,-1,-1,
						 -1,2,-1,
						 -1,-1,2 }; //135�� ����ũ

	//����ũ ���� 0�� �ǹǷ� 1�� ����ȭ�ϴ� ������ �ʿ� ����

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dstData[y * width + x] = (abs(myKernelCon3x3(srcData, kernelX, x, y, width, height)) + abs(myKernelCon3x3(srcData, kernelY, x, y, width, height))) / 2;
			//�� ���� ����� ���밪 �� ���·� ���� ��� ����

		}
	}
	return dstImg;
}

Mat mySampling(Mat srcImg)
{
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;

	Mat dstImg(height, width, CV_8UC1);

	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dstData[y * width + x] = srcData[(y * 2) * (width * 2) + (x * 2)];
			//2�� �������� �ε��� �� ū ������ ���� ���� ������ �� ����
		}
	}

	return dstImg;
}

vector <Mat> myGaussianPyramid(Mat srcImg)
{
	vector<Mat> Vec;

	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++)
	{
		srcImg = mySampling(srcImg);
		srcImg = myGaussianfilter(srcImg);
		Vec.push_back(srcImg); // vector �����̳ʿ� �ϳ��� ó�� ����� ����
	}
	return Vec;
	
}