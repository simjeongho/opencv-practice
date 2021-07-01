#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

double gaussian2D(float c, float r, double sigma);
void myGaussian(const Mat& src_img, Mat& dst_img, Size size);
void myKernelConv(const Mat& src_img, Mat dst_img, const Mat& kn);
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx();
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void doBilateralEx();
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
float distance(int x, int y, int i, int j);
double gaussian(float x, double sigma);

int main()
{
	//1�� median filter
	Mat salt_pepper = imread("salt_pepper2.png", 0);
	Mat result;
	imshow("salt peper", salt_pepper);
	doMedianEx();
	medianBlur(salt_pepper, result, 5);
	imshow("result", result);

	//2�� 
	clock_t start, end;
	double result1;
	int LTH = 10;
	int HTH = 20;
	Mat edge_test = imread("edge_test.jpg", 0);
	Mat edge_result;
	start = clock();
	Canny(edge_test, edge_result, LTH, HTH, 3, false);
	imshow("edge result", edge_result);
	end = clock();
	result1= (double)(end - start);
	cout << "Low Threshhold :" << LTH << " High Threshhold : " << HTH << endl;
	cout << endl << "Canny edge detection ���� �ð� " <<result1 << endl;


	//3��
	Mat rock = imread("rock.png", 0);
	imshow("rock", rock);
	doBilateralEx();

	waitKey(0);
}

double gaussian2D(float c, float r, double sigma)
{
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}
void myGaussian(const Mat& src_img, Mat& dst_img, Size size)
{
	//Ŀ�� ����
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 2.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++)
	{
		for (int r = 0; r < kn.rows; r++)
		{
			kn_data[r * kn.cols + c] = (float)gaussian2D((float)(c - kn.cols / 2), (float)(r - kn.rows / 2), sigma);
		}
	}
	//Ŀ�� ������� ����
	myKernelConv(src_img, dst_img, kn);
}

void myKernelConv(const Mat& src_img, Mat dst_img, const Mat& kn)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols;
	int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2;
	int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	//�ʼ� �ε���(�����ڸ� ����)
	for (int c = rad_w + 1; c < wd - rad_w; c++)
	{
		for (int r = rad_h + 1; r < hg - rad_h; r++)
		{
			tmp = 0.f;
			sum = 0.f;
			//<Ŀ�� �ε���>
			for (int kc = -rad_w; kc <= rad_w; kc++)
			{
				for (int kr = -rad_h; kr <= rad_h; kr++)
				{
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum; //����ȭ �� overflow ���� abs���밪
			else tmp = abs(tmp);

			if (tmp > 255.f)tmp = 255.f; // overflow ����

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1); //����ũ 0���� ����

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height; //Ŀ�� ����ũ
	int rad_w = kwd / 2; int rad_h = khg / 2; // Ŀ���� ��
	int numcenter = kwd*khg / 2;
	uchar* src_data = (uchar*)src_img.data; // ������ ������
	uchar* dst_data = (uchar*)dst_img.data; //����ũ�� ������

	float* table = new float[khg * kwd](); // Ŀ�� ���̺� �����Ҵ�
	float tmp;
	cout << "for �� ��" << endl;
	cout << rad_w << "rad_w" << endl;
	cout << rad_h << "rad_h" << endl;
	cout << kwd << " kwd " << khg << " khg" << endl;
	cout << wd << " wd " << hg << " hg " << endl;
	//�ȼ� �ε���(�����ڸ� ����) ���� �ڸ� 
	for (int c = rad_w+1; c < wd - rad_w; c++)
	{
		for (int r = rad_h+1; r < hg - rad_h; r++)
		{
			for (int i = 0; i < kwd; i++) {
				for (int j = 0; j < khg; j++) {
					table[5 * i + j] = src_data[wd * (c - rad_w + j) + (r - rad_h + i)];
				}
			}

			for (int max = 0; max < kwd * khg -1; max++)
			{
				for (int i = 0; i < kwd * khg - 1; i++)
				{
					if (table[i] > table[i + 1]) //ū ������ ����
					{
						tmp = table[i];
						table[i] = table[i + 1];
						table[i + 1] = tmp; // ���̺� �ȼ� ����
						
					}
					
				}
				
			} // ���� ��
			
			dst_data[(c-1) * wd + r-1] = table[kwd *khg /2 +1]; // ���� ū ���� �߰������� ��ü
		}
	}
	//for (int r = 0; r < hg; r++)
	//{
	//	for (int c = 0; c < wd; c++)
	//	{
	//		dst_data[r* wd +c] = table[r * wd + c]; //����ũ �̹����� ����
	//	}
	//}
	
	delete []table; // �����Ҵ� ���� 
}

void doMedianEx()
{
	cout << " ---doMediaEx() --- \n" << endl;

	//<�Է�>
	Mat src_img = imread("salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");
	//imshow("this is pepper", src_img);

	//<Median ���͸� ����>
	Mat dst_img;
	cout << "���� ��" << endl;
	myMedian(src_img, dst_img, Size(5, 5));

	//���
	Mat result_img;
	hconcat(src_img, dst_img, result_img); // �� ���� �̹����� ���η� �ٿ� ���
	imshow("doMedianEx()", result_img);
	//waitKey(0);
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s)
{
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;

	//�ȼ� �ε���(�����ڸ� ����)>
	for (int c = radius + 1; c < hg - radius; c++)
	{
		for (int r = radius + 1; r < wh - radius; r++)
		{
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			//ȭ�Һ� Bilateral ��� ����
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1); // Mat type ��ȯ
}

void doBilateralEx()
{
	cout << " --- do BilateralEx()  --- \n" << endl;

	// < �Է� >
	Mat src_img = imread("rock.png", 0);
	//Mat src_img;
	//resize(src_img1, src_img, Size(200, 200));
	Mat dst_img1_2;
	Mat dst_img25_2;
	Mat dst_img1000_2;
	Mat dst_img1_6;
	Mat dst_img25_6;
	Mat dst_img1000_6;
	Mat dst_img1_18;
	Mat dst_img25_18;
	Mat dst_img1000_18;
	if (!src_img.data) printf("No image data \n");

	// < Bilateral ���͸� ����>
	myBilateral(src_img, dst_img1_2, 5, 0.1, 1);
	myBilateral(src_img, dst_img25_2, 5, 0.25, 1);
	myBilateral(src_img, dst_img1000_2, 5, 1000, 1);
	myBilateral(src_img, dst_img1_6, 5, 0.1, 100);
	myBilateral(src_img, dst_img25_6, 5, 0.25, 100);
	myBilateral(src_img, dst_img1000_6, 5, 1000, 100);
	myBilateral(src_img, dst_img1_18, 5, 0.1, 100000000);
	myBilateral(src_img, dst_img25_18, 5, 0.25, 1000000000);
	myBilateral(src_img, dst_img1000_18, 5,1000, 1000000000);
	//<���>
	Mat result_img2_0;
	Mat result_img2_1;
	Mat result_img6_0;
	Mat result_img6_1;
	Mat result_img18_0;
	Mat result_img18_1;
	imshow("1_2", dst_img1_2);
	imshow("1_18", dst_img1_18);
	Mat result_img2;
	Mat complete;
	hconcat(dst_img1_2, dst_img25_2, result_img2_0);
	hconcat(result_img2_0, dst_img1000_2, result_img2_1);

	hconcat(dst_img1_6, dst_img25_6, result_img6_0);
	hconcat(result_img6_0, dst_img1000_6, result_img6_1);

	hconcat(dst_img1_18, dst_img25_18, result_img18_0);
	hconcat(result_img18_0, dst_img1000_18, result_img18_1);
	imshow("s = 1", result_img2_1);
	imshow("s = 100", result_img6_1);
	imshow("s = 10000000", result_img18_1);
	vconcat(result_img2_1, result_img6_1, result_img2);
	vconcat(result_img2, result_img18_1, complete);
	imshow("doBilateralEx()", result_img2_1);
	imshow("complete", complete);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s)
{
	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;

	//<Ŀ�� �ε���>
	for (int kc = -radius; kc <= radius; kc++)
	{
		for (int kr = -radius; kr <= radius; kr++)
		{
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			//range calc
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			//spatial calc
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; // ����ȭ
}

double gaussian(float x, double sigma)
{
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI + pow(sigma, 2)); //exp �����Լ�
}
//double gaussian2D(float c, float r, double sigma)
//{
//	return exp(-(pow(c, 2) + pow(r, 2) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
//}
float distance(int x, int y, int i, int j)
{
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2))); // pow n����, sqrt��Ʈ
}

