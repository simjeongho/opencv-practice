#include <iostream>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;
using namespace cv;
void readImagesAndTimes(vector<Mat>& images, vector<float>& times);
Mat GetHistogram(Mat& src);
void main()
{
	//영상 노출 시간 불러오기

	cout << "Reading images and exposure times..." << endl;
	vector<Mat> images;
	vector<float>times;
	readImagesAndTimes(images, times);
	cout << "finished" << endl;

	cout << "Aligning images ... " << endl;
	Ptr<AlignMTB> alignMTB = createAlignMTB();
	alignMTB->process(images, images);

	//Camera response function(CRF)복원
	cout << "Calculating Camera Response Function ... " << endl;
	Mat responseDebevec;
	Ptr<CalibrateDebevec>calibrateDebevec = createCalibrateDebevec();
	calibrateDebevec->process(images, responseDebevec, times);
	cout << "-------CRF -------" << endl;
	cout << responseDebevec << endl;

	//24bit 표현 범위로 이미지 병합
	cout << "Merging images into one HDR image ..." << endl;
	Mat hdrDebevec;
	Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
	mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
	imwrite("hdrDebevec.hdr", hdrDebevec);
	cout << "saved hdrDebevec.hdr" << endl;

	//cout << "Merging using Exposure Fusion ... " << endl;
	//Mat hdrDebevec;
	//Ptr<MergeMertens> mergeMertens = createMergeMertens();
	//mergeMertens->process(images, hdrDebevec);

		//<< Drage 톤맵>>
	cout << "Tonemaping using Drage's method ...";
	Mat IdrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
	tonemapDrago->process(hdrDebevec, IdrDrago);
	IdrDrago = 3 * IdrDrago;
	imwrite("myIdr-Drago.jpg", IdrDrago * 255);
	cout << "saved Idr-Drago.jpg" << endl;

	//<Reinhard톤맵>>
	cout << "Tonemaping using Reinhard's method ...";
	Mat IdrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
	tonemapReinhard->process(hdrDebevec, IdrReinhard);
	imwrite("myIdr-Reinhard.jpg", IdrReinhard * 255);
	cout << "saved Idr-Reinhard.jpg" << endl;

	//<<Mantiuk 톤맵>>
	cout << "Tonemaping using Mantiuk's method ...";
	Mat IdrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
	tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
	IdrMantiuk = 3 * IdrMantiuk;
	imwrite("myIdr-Mantiuk.jpg", IdrMantiuk * 255);
	cout << "saved Idr-Mantiuk.jpg" << endl;
	
	Mat img_0 = imread("my1.jpg", 0);
	Mat histo0 = GetHistogram(img_0);
	imshow("hist0", histo0);

	Mat img_1 = imread("my2.jpg", 0);
	Mat histo1 = GetHistogram(img_1);
	imshow("hist1", histo1);

	Mat img_2 = imread("my3.jpg", 0);
	Mat histo2 = GetHistogram(img_2);
	imshow("hist2", histo2);

	Mat img_3 = imread("my4.jpg", 0);
	Mat histo3 = GetHistogram(img_3);
	imshow("hist3", histo3);

	Mat drago = imread("myIdr-Drago.jpg", 0);
	imshow("drago" , drago);
	Mat histodrago = GetHistogram(drago);
	imshow("histo drago", histodrago);

	Mat Reinhard = imread("myIdr-Reinhard.jpg", 0);
	imshow("Reinhard", Reinhard);
	Mat histoReinhard = GetHistogram(Reinhard);
	imshow("histo Reinhard", histoReinhard);

	Mat Mantiuk = imread("myIdr-Mantiuk.jpg", 0);
	imshow("Mantiuk", Mantiuk);
	Mat histoMantiuk = GetHistogram(Mantiuk);
	imshow("histo Mantiuk", histoMantiuk);

	waitKey(0);
}

//void exposure()
//{
//	//영상 노출 시간 불러오기
//
//	cout << "Reading images and exposure times..." << endl;
//	vector<Mat> images;
//	vector<float>times;
//	readImagesAndTimes(images, times);
//	cout << "finished" << endl;
//}

void readImagesAndTimes(vector<Mat>& images, vector<float>& times)
{
	int numImages = 4;
	static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
	times.assign(timesArray, timesArray + numImages);
	//static const char* filenames[] = { "img_0.033.jpg" , "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg" };
	static const char* filenames[] = { "my1.jpg" , "my2.jpg", "my3.jpg", "my4.jpg" };
	for (int i = 0; i < numImages; i++)
	{
		Mat im = imread(filenames[i]);
		images.push_back(im);
	}
}

Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	
	//히스토그램 계산
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat()); // 정규화

	for (int i = 1; i < number_bins; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i - 1))), Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}

//void Aligning()
//{
//	cout << "Aligning images ... " << endl;
//	Ptr<AlignMTB> alignMTB = createAlignMTB();
//	alignMTB->process(images, images);
//}

//void reCRF()
//{
//	//Camera response function(CRF)복원
//	cout << "Calculating Camera Response Function ... " << endl;
//	Mat responseDebevec;
//	Ptr<CalibrateDebevec>calibrateDebevec = createCalibrateDebevec();
//	calibrateDebevec->process(images, responseDebevec, times);
//	cout << "-------CRF -------" << endl;
//	cout << responseDebevec << endl;
//}

//void two_fourbit()
//{
//	//24bit 표현 범위로 이미지 병합
//	cout << "Merging images into one HDR image ..." << endl;
//	Mat hdrDebevec;
//	Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
//	mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
//	imwrite("hdrDebevec.hdr", hdrDebevec);
//	cout << "saved hdrDebevec.hdr" << endl;
//
//	//cout << "Merging using Exposure Fusion ... " << endl;
//	//Mat hdrDebevec;
//	//Ptr<MergeMertens> mergeMertens = createMergeMertens();
//	//mergeMertens->process(images, hdrDebevec);
//}
//
//void mytonemap()
//{
//	//<< Drage 톤맵>>
//	cout << "Tonemaping using Drage's method ...";
//	Mat IdrDrago;
//	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
//	tonemapDrago->process(hdrDebevec, IdrDrago);
//	IdrDrago = 3 * IdrDrago;
//
//}