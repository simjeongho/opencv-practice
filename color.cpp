#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
void CvColorModels(Mat bgr_img);
Mat GetYCbCr(Mat src_img);
//Mat GetHSV(Mat src_img);
Mat MyKMeans(Mat src_img, int n_cluster);
void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptlnClusters);
void findAssociatedCluster(Mat imgInput, int n_cluster, vector<Scalar> clustersCenters, vector<vector<Point>>& ptlnClusters);
double computeColorDistance(Scalar pixel, Scalar clusterPixel);
double adjustClusterCenters(Mat src_img, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>> ptlnClusters, double& oldCenter, double newCenter);
Mat applyClusterToImage(Mat src_img, int n_cluster, vector<vector<Point>> ptlnClusters, vector<Scalar> clustersCenters);
Mat applyFinalClusterToImage(Mat src_img, int n_cluster, vector<vector<Point>> ptlnClusters, vector<Scalar> clustersCenters);
float min1(float b, float g, float r);
float max1(float b, float g, float r);
Mat MyBgr2Hsv(Mat src_img);
int main()
{
	Mat fruit = imread("fruit1.jpeg", 1);
	MyKMeans(fruit, 7);

	Mat hsvfruit;
	cvtColor(fruit, hsvfruit, cv::COLOR_BGR2HSV);
	imshow("hsv ", hsvfruit);
	Mat hsv = MyBgr2Hsv(fruit);
	imshow("my hsv", hsv);
	waitKey(0);

}

void CvColorModels(Mat bgr_img)
{
	Mat gray_img, rgb_img, hsv_img, yuv_img, xyz_img;
	cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
	cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
	cvtColor(bgr_img, hsv_img, cv::COLOR_BGR2HSV);
	cvtColor(bgr_img, yuv_img, cv::COLOR_BGR2YCrCb);
	cvtColor(bgr_img, xyz_img, cv::COLOR_BGR2XYZ);

	Mat print_img;
	bgr_img. copyTo(print_img);
	cvtColor(print_img, gray_img, cv::COLOR_GRAY2BGR);
	hconcat(print_img, gray_img, print_img);
	hconcat(print_img, rgb_img, print_img);
	hconcat(print_img, hsv_img, print_img);
	hconcat(print_img, yuv_img, print_img);
	hconcat(print_img, xyz_img, print_img);

	imshow("results", print_img);
	imwrite("CvColorModels.png", print_img);

	waitKey(0);

}

Mat GetYCbCr(Mat src_img)
{
	double b, g, r, y, cb, cr;
	Mat dst_img;
	src_img.copyTo(dst_img);

	//<ȭ�� �ε���>
	for (int row = 0; row < dst_img.rows; row++)
	{
		for(int col = 0; col < dst_img.cols; col++)
		{
			//<BGR ���>
			//OpenCV�� Mat�� BGR�� ������ ������ ����
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			//<���� ��ȯ ���>
			//��Ȯ�� ����� ���� double �ڷ��� ���
			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			//<�����÷ο� ����>
			y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			//<��ȯ�� ���� ����>
			//double �ڷ����� ���� ���� �ڷ������� ��ȯ
			dst_img.at<Vec3b>(row, col)[0] = (uchar)y;
			dst_img.at<Vec3b>(row, col)[1] = (uchar)cb;
			dst_img.at<Vec3b>(row, col)[2] = (uchar)cr;
		}
	}
	return dst_img;
}
Mat MyBgr2Hsv(Mat src_img)
{
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++)
	{
		for (int x = 0; x < src_img.cols; x++)
		{
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];
		
			double max = max1(b,g,r);
			double min = min1(b, g, r);
			double z = max - min;
			v = max * 255;
			if (v == 0.0) s == 0.0;
			else {
				s = z / max * 255; // 255�� �������� Ű����
			}
			if (s == 0.0) h = 0.0;
			else {
				if (max == r)
				{
					h = 60 * (g - b) / z;
					//h = h / 2;
				}
				else if (max == g)
				{
					h = 60 * (2 + (b - r)) / z ;
					//h = h / 2;
				}
				else
				{
					h = 60 * (4 + (r - g)) / z ;
					//h = h / 2;
				}
			}
			if (h < 0)
			{
				h = h + 180;
			}
			else h = h;
			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;

		}
	}
	return dst_img;
}
float max1(float b, float g, float r)
{
	float max;
	if (r > g)
	{
		max = r;
	}
	else max = g;
	if (max > b) max = max;
	else max = b;

	return max;
}
float min1(float b, float g, float r)
{
	float min;
	if (r > g) min = g;
	else min = r;
	if (min > b) min = b;
	else min = min;
	return min;
}
//Mat GetHSV(Mat src_img)
//{
//
//}
Mat MyKMeans(Mat src_img, int n_cluster)
{
	vector <Scalar> clustersCenters; // ���� �߾Ӱ� ����
	vector <vector<Point>> ptlnClusters; // ������ ��ǥ ���� ���� ����  ������ �Ǻ��ϰ� �� ȭ���� �ε����� ��´�.
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter; // ���� ������ ��ȭ��

	//<�ʱ⼳��>
	//���� �߾Ӱ��� �������� �Ҵ� �� ���� �� ��ǥ���� ������ ���� �Ҵ�
	createClustersInfo(src_img, n_cluster, clustersCenters, ptlnClusters);

	//<�߾Ӱ� ���� �� ȭ�Һ� ���� �Ǻ�>
	//�ݺ����� ������� ���� �߾Ӱ� ����
	//������ �Ӱ谪 ���� ���� ������ ��ȭ�� ���� �� ���� �ݺ�
	while (diffChange > threshold)//�߾Ӱ� ����
	{
		//<�ʱ�ȭ>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++)
		{
			ptlnClusters[k].clear();
		}

		//������ ���� �߾Ӱ��� �������� ���� Ž��>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptlnClusters);

		//���� �߾Ӱ� ����
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptlnClusters, oldCenter, newCenter);
		
	}

	//���� �߾Ӱ����θ� �̷���� ���� ����>
	Mat dst_img = applyFinalClusterToImage(src_img, n_cluster, ptlnClusters, clustersCenters);
	imshow("result", dst_img);
	return dst_img;
}

void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptlnClusters)
{
	RNG random(cv::getTickCount()); // opencv���� ������ ���� �����ϴ� �Լ� ��ǥ�� �������� �����ؼ� �� ��ǥ�� ���� �߾Ӱ����� �������ش�.

	for (int k = 0; k < n_cluster; k++) //���ɿ��� ����
	{
		//���� �� ���
		//<������ ��ǥ ȹ��>
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		//<������ ��ǥ�� ȭ�Ұ����� ������ �߾Ӱ� ����>
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point> ptlnClusterK;
		ptlnClusters.push_back(ptlnClusterK);//�߾Ӱ��� ���Ϳ� �����Ѵ�. 
	}
}

void findAssociatedCluster(Mat imgInput, int n_cluster, vector<Scalar> clustersCenters, vector<vector<Point>>& ptlnClusters)
{
	for (int r = 0; r < imgInput.rows; r++)
	{
		for (int c = 0; c < imgInput.cols; c++)
		{
				double minDistance = INFINITY;
				int closestClusterIndex = 0;
				Scalar pixel = imgInput.at<Vec3b>(r, c);

				for (int k = 0; k < n_cluster; k++)
				{
					//������ ���
					//<�� ���� �߾Ӱ����� ���̸� ���>
					Scalar clusterPixel = clustersCenters[k];
					double distance = computeColorDistance(pixel, clusterPixel);

					//<���̰� ���� ���� �������� ��ǥ�� ������ �Ǻ�> //�߾Ӱ���� ��� ȭ�ҵ� ���� �Ÿ� ��� 
					if (distance < minDistance)
					{
						minDistance = distance;
						closestClusterIndex = k;
					}
				}

				//<��ǥ ����>
				ptlnClusters[closestClusterIndex].push_back(Point(c, r)); // ���Ϳ� ���� 
	
		}
	}
}

double computeColorDistance(Scalar pixel, Scalar clusterPixel)
{
	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));
	//Euclidian distance

	return distance;
}

double adjustClusterCenters(Mat src_img, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>> ptlnClusters, double& oldCenter, double newCenter)
{// �����ͷ� ���� �޾ƿ��� �� ���� ���� ����Ű�� ���ҵ� ������ ������ �ϳ��ۿ� ���ؼ� ���� ����
	double diffChange;

	for (int k = 0; k < n_cluster; k++)
	{
		//���� �� ���
		vector<Point>ptlnCluster = ptlnClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//<��հ� ���>
		for (int i = 0; i < ptlnCluster.size(); i++)
		{
			Scalar pixel = src_img.at<Vec3b>(ptlnCluster[i].y, ptlnCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptlnCluster.size(); // ����� ���Ѵ�. 
		newGreen /= ptlnCluster.size();
		newRed /= ptlnCluster.size();

		//<����� ��հ����� ���� �߾Ӱ� ��ü>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//  ��� ������ ���� ��հ��� ���� ���
		clustersCenters[k] = newPixel;
	}

	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//��� ������ ���� ��հ� ��ȭ�� ���

	oldCenter = newCenter;

	return diffChange;
}

Mat applyClusterToImage(Mat src_img, int n_cluster, vector<vector<Point>> ptlnClusters, vector<Scalar> clustersCenters)
{
	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++)
	{
		//��� ������ ���� ����
		vector<Point>ptlnCluster = ptlnClusters[k]; // ������ ��ǥ��
		for (int j = 0; j < ptlnCluster.size(); j++)
		{
			//������ ��ǥ ��ġ�� �ִ� ȭ�� ���� �ش� ���� �߾Ӱ����� ��ü
			dst_img.at<Vec3b>(ptlnCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptlnCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptlnCluster[j])[2] = clustersCenters[k].val[2];
			cout << clustersCenters[k].val[0] << endl;
			cout << clustersCenters[k].val[1] << endl;
			cout << clustersCenters[k].val[2] << endl;
		}
	}
	return dst_img;
}

Mat applyFinalClusterToImage(Mat src_img, int n_cluster, vector<vector<Point>> ptlnClusters, vector<Scalar> clustersCenters)
{
	RNG random(cv::getTickCount());
	Mat dst_img(src_img.size(), src_img.type());
	int rd, rd1, rd2 = 0;
	
	double *rdnumber = new double[n_cluster * 3];
	for (int i = 0; i < n_cluster; i++)
	{
		rd = rand() % 255;
		rd1 = rand() % 255;
		rd2 = rand() % 255;
		rdnumber[i] = rd;
		rdnumber[i + n_cluster] = rd1;
		rdnumber[i + (n_cluster * 2)] = rd2;
		
	}	
	for (int k = 0; k < n_cluster; k++)
	{
		vector<Point>ptlnCluster = ptlnClusters[k]; // ������ ��ǥ��
		for (int j = 0; j < ptlnCluster.size(); j++)
		{
			//������ ��ǥ ��ġ�� �ִ� ȭ�� ���� �ش� ���� �߾Ӱ����� ��ü
			dst_img.at<Vec3b>(ptlnCluster[j])[0] = rdnumber[k];
			dst_img.at<Vec3b>(ptlnCluster[j])[1] = rdnumber[k+n_cluster];
			dst_img.at<Vec3b>(ptlnCluster[j])[2] = rdnumber[k+(n_cluster*2)];
			
		}
	}
	return dst_img;
}

