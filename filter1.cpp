#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
Mat doDft(Mat srcImg); // discrete fourier transform
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat centralize(Mat complex);
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);
Mat doLPF(Mat srcImg);
Mat doHPF(Mat srcImg);
Mat doBPF(Mat srcImg);
Mat doFPF(Mat srcImg);
Mat doSPF(Mat srcImg);
int myKernelCon3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height); //convolution
Mat mySobelFilter(Mat srcImg);

int main()
{ // 1��
	Mat einstein = imread("img1_5.jpg", 0);
	//imshow("einstein", einstein);
	Mat lpeinstein= doLPF(einstein);
	//imshow("lpf", lpeinstein);
	Mat bfeinstein = doBPF(einstein);
	imshow("bandpass", bfeinstein);
	Mat heinstein = doHPF(einstein);
	//imshow("high", heinstein);

	//2��
	Mat src2 = imread("img2_5.jpg " , 0);
	Mat src3 = imread("img2_5.jpg", 0);
	Mat sobel = mySobelFilter(src2);
	imshow("sobel", sobel);
	imshow("src2", src2);
	//fourier transform
	Mat ftshirt = doDft(src3);
	Mat centershirt = centralize(ftshirt);
	Mat magshirt = getMagnitude(centershirt);
	Mat normalmag = myNormalize(magshirt);
	imshow("magnitude", normalmag);

	//get phasor center -> myNormalize
	Mat phashirt = getPhase(centershirt);
	Mat normalpha = myNormalize(phashirt);
	imshow("phasor", normalpha);
	
	Mat sobelimg2 = doSPF(src2);
	imshow("sobelimg2", sobelimg2);

	//Mat complexshirt = setComplex(normalmag, normalpha);
	//Mat idftshirt = doIdft(complexshirt);

	//imshow("dft", idftshirt);

	//3��
	Mat flicker = imread("img3_5.jpg", 0);
	imshow("flicker", flicker);
	Mat ftflicker = doDft(flicker);
	Mat centerflicker = centralize(ftflicker);
	Mat magflicker = getMagnitude(centerflicker);
	Mat normalflicker = myNormalize(magflicker);
	imshow("flickermag", normalflicker);
	Mat lpflicker = doFPF(flicker);
	imshow("fpflicker", lpflicker);


	waitKey(0);
}

Mat doDft(Mat srcImg) // discrete fourier transform
{
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg)
{
	Mat planes[2];
	split(complexImg, planes);
	//�Ǽ���, ����� �и�

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//magnitude ���
	//log(1+sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src)
{
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
} // normalize

Mat getPhase(Mat complexImg)
{
	Mat planes[2];
	split(complexImg, planes);
	//�Ǽ��� , ����� �и�

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);

	return phaImg;
}


Mat centralize(Mat complex)
{
	Mat planes[2];

	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;// �ӽ� ����

	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);
	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;

} // ��ǥ�� �߾� �̵�;

Mat setComplex(Mat magImg, Mat phaImg)
{
	exp(magImg, magImg);

	magImg -= Scalar::all(1);
	//magnitude ����� �ݴ�� ����

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	//�� ��ǥ�� -> ���� ��ǥ��(������ ũ��κ��� 2���� ��ǥ)

	Mat complexImg;
	merge(planes, 2, complexImg);
	//�Ǽ���, ����� ��ü
	return complexImg; // �ٽ� complexImg ���
}

Mat doIdft(Mat complexImg)
{
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT�� �̿��� ���� ���� ���

	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);

	return dstImg;
}

Mat doLPF(Mat srcImg)
{
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	//imshow("LPFmask", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}
Mat doSPF(Mat srcImg)
{
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);
	int radius = 150;
	int width = magImg.cols / 3 +10;
	int height = magImg.rows / 3 + 10;
	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	/*circle(maskImg, Point(maskImg.cols / 4,  maskImg.rows /4),radius , Scalar::all(1), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 4 * 3, maskImg.rows / 4), radius, Scalar::all(1), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 4 * 3, maskImg.rows / 4 * 3), radius, Scalar::all(1), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 4  , maskImg.rows / 4 * 3), radius, Scalar::all(1), -1, -1, 0);*/
	//rectangle(maskImg, Rect(Point(magImg.cols / 2 - 100, magImg.rows/2 +100), Point(magImg.cols / 2 + 100, magImg.rows/2 -100)), Scalar::all(0), -1, 8, 0);
	//rectangle(maskImg, Rect(Point(0, magImg.rows-1), Point(magImg.cols / 3, magImg.rows / 3 * 2)), Scalar::all(0), -1, 8, 0); // ���� ��
	//rectangle(maskImg, Rect(Point(magImg.cols / 3 * 2, magImg.rows - 1), Point(magImg.cols - 1, magImg.rows / 3 * 2)), Scalar::all(0), -1, 8, 0); // ������ ��
	//rectangle(maskImg, Rect(magImg.cols / 3 , magImg.rows / 3 , width , height) , Scalar::all(0), -1, 8 , 0); //�߾�
	//rectangle(maskImg, Rect( 0 , 0 , width, height), Scalar::all(0), -1, 8, 0); // ���� �Ʒ�
	//rectangle(maskImg, Rect(magImg.cols / 3 * 2 , 0  , width, height), Scalar::all(0), -1, 8, 0); // ������ �Ʒ�
	rectangle(maskImg, Rect(magImg.cols / 3 , magImg.rows/3*2, width, height), Scalar::all(1), -1, 8, 0); // 1�� �� �߾�
	rectangle(maskImg, Rect(0, magImg.rows / 3 , width, height), Scalar::all(1), -1, 8, 0); //scalar 1�� �� ����
	rectangle(maskImg, Rect(magImg.cols / 3 * 2, magImg.rows / 3, width, height), Scalar::all(1), -1, 8, 0); //scalar 1�� �� ������
	rectangle(maskImg, Rect(magImg.cols / 3, 0 ,  width, height), Scalar::all(1), -1, 8, 0); // scalar 1�� �� �Ʒ�
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	imshow("SPFmask", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

Mat doFPF(Mat srcImg)
{
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);
	
	
	int width = 100;
	int height = 200;
	//<FPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	imshow("flickerfiltermag", magImg);
	Mat maskImg = Mat::ones(magImg.size(), CV_32F);

	rectangle(maskImg, Rect(magImg.cols / 2 - 1, magImg.rows / 2 - 7, 2, 3), Scalar::all(0), -1, 8, 0);
	rectangle(maskImg, Rect(magImg.cols /2 -1, magImg.rows / 2 + 7, 2, 3), Scalar::all(0), -1, 8, 0);

	//for (int i = 0; i < 375; i+=15)
	//{
	//	rectangle(maskImg, Rect(magImg.cols / 2 -15, 0 + i, 14, 2), Scalar::all(0), -1, 8, 0);
	//}
	//for (int i = 405; i < 1000; i += 15)
	//{
	//	rectangle(maskImg, Rect(magImg.cols / 2 - 15, 0 + i, 14, 2), Scalar::all(0), -1, 8, 0);
	//}
	//for (int i = 0; i < 375; i += 15)
	//{
	//	rectangle(maskImg, Rect(magImg.cols / 2 +1, 0 + i, 14, 2), Scalar::all(0), -1, 8, 0);
	//}
	//for (int i = 405; i < 1000; i += 15)
	//{
	//	rectangle(maskImg, Rect(magImg.cols / 2 +1, 0 + i, 14, 2), Scalar::all(0), -1, 8, 0);
	//}
	imshow("maskrect", maskImg);
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	imshow("maskFPF", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}
Mat doBPF(Mat srcImg)
{

	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<BPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 100, Scalar::all(1), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0);
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	imshow("BPFmask", magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);
	return myNormalize(dstImg);

}
Mat doHPF(Mat srcImg)
{
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);
	
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	//imshow("HPFmask", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);
	return myNormalize(dstImg);
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
Mat mySobelFilter(Mat srcImg)
{
	int kernelX[3][3] = { -1, 0,1,
						-2,0,2,
						-1 ,0,1 }; // x�� ����ũ
	int kernelY[3][3] = { 1,2,1,
						 0,0,0,
						 -1,-2,-1 }; //y�� ����ũ

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