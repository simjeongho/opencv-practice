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

	//<화소 인덱싱>
	for (int row = 0; row < dst_img.rows; row++)
	{
		for(int col = 0; col < dst_img.cols; col++)
		{
			//<BGR 취득>
			//OpenCV의 Mat은 BGR의 순서를 가짐에 유의
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			//<색상 변환 계산>
			//정확한 계산을 위해 double 자료형 사용
			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			//<오버플로우 방지>
			y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			//<변환된 색상 대입>
			//double 자료형의 값을 본래 자료형으로 변환
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
				s = z / max * 255; // 255로 스케일을 키워줌
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
	vector <Scalar> clustersCenters; // 군집 중앙값 벡터
	vector <vector<Point>> ptlnClusters; // 군집별 좌표 벡터 이중 벡터  군집을 판별하고 각 화소의 인덱스를 담는다.
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter; // 군집 조정의 변화량

	//<초기설정>
	//군집 중앙값을 무작위로 할당 및 군집 별 좌표값을 저장할 벡터 할당
	createClustersInfo(src_img, n_cluster, clustersCenters, ptlnClusters);

	//<중앙값 조정 및 화소별 군집 판별>
	//반복적인 방법으로 군집 중앙값 조정
	//설정한 임계값 보다 군집 조정의 변화가 작을 때 까지 반복
	while (diffChange > threshold)//중앙값 조정
	{
		//<초기화>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++)
		{
			ptlnClusters[k].clear();
		}

		//현재의 군집 중앙값을 기준으로 군집 탐색>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptlnClusters);

		//군집 중앙값 조절
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptlnClusters, oldCenter, newCenter);
		
	}

	//군집 중앙값으로만 이루어진 영상 생성>
	Mat dst_img = applyFinalClusterToImage(src_img, n_cluster, ptlnClusters, clustersCenters);
	imshow("result", dst_img);
	return dst_img;
}

void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptlnClusters)
{
	RNG random(cv::getTickCount()); // opencv에서 무작위 값을 설정하는 함수 좌표를 무작위로 설정해서 그 좌표의 색을 중앙값으로 설정해준다.

	for (int k = 0; k < n_cluster; k++) //관심영역 설정
	{
		//군집 별 계산
		//<무작위 좌표 획득>
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		//<무작위 좌표의 화소값으로 군집별 중앙값 설정>
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point> ptlnClusterK;
		ptlnClusters.push_back(ptlnClusterK);//중앙값을 벡터에 저장한다. 
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
					//군집별 계산
					//<각 군집 중앙값과의 차이를 계산>
					Scalar clusterPixel = clustersCenters[k];
					double distance = computeColorDistance(pixel, clusterPixel);

					//<차이가 가장 적은 군집으로 좌표의 군집을 판별> //중앙값들과 모든 화소들 간의 거리 계산 
					if (distance < minDistance)
					{
						minDistance = distance;
						closestClusterIndex = k;
					}
				}

				//<좌표 저장>
				ptlnClusters[closestClusterIndex].push_back(Point(c, r)); // 벡터에 저장 
	
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
{// 포인터로 값을 받아오는 건 값을 직접 가르키는 역할도 하지만 리턴을 하나밖에 못해서 직접 접근
	double diffChange;

	for (int k = 0; k < n_cluster; k++)
	{
		//군집 별 계산
		vector<Point>ptlnCluster = ptlnClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//<평균값 계산>
		for (int i = 0; i < ptlnCluster.size(); i++)
		{
			Scalar pixel = src_img.at<Vec3b>(ptlnCluster[i].y, ptlnCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptlnCluster.size(); // 평균을 구한다. 
		newGreen /= ptlnCluster.size();
		newRed /= ptlnCluster.size();

		//<계산한 평균값으로 군집 중앙값 대체>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//  모든 군집에 대한 평균값도 같이 계산
		clustersCenters[k] = newPixel;
	}

	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//모든 군집에 대한 평균값 변화량 계산

	oldCenter = newCenter;

	return diffChange;
}

Mat applyClusterToImage(Mat src_img, int n_cluster, vector<vector<Point>> ptlnClusters, vector<Scalar> clustersCenters)
{
	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++)
	{
		//모든 군집에 대해 수행
		vector<Point>ptlnCluster = ptlnClusters[k]; // 군집별 좌표들
		for (int j = 0; j < ptlnCluster.size(); j++)
		{
			//군집별 좌표 위치에 있는 화소 값을 해당 군집 중앙값으로 대체
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
		vector<Point>ptlnCluster = ptlnClusters[k]; // 군집별 좌표들
		for (int j = 0; j < ptlnCluster.size(); j++)
		{
			//군집별 좌표 위치에 있는 화소 값을 해당 군집 중앙값으로 대체
			dst_img.at<Vec3b>(ptlnCluster[j])[0] = rdnumber[k];
			dst_img.at<Vec3b>(ptlnCluster[j])[1] = rdnumber[k+n_cluster];
			dst_img.at<Vec3b>(ptlnCluster[j])[2] = rdnumber[k+(n_cluster*2)];
			
		}
	}
	return dst_img;
}

