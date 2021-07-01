#include <iostream>
#include <time.h>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching.hpp> // Stitching
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
using namespace
cv::xfeatures2d;

void ex_panorama_simple();
Mat findbook(Mat img_l, Mat img_r, int thresh_dist, int min_matches);
Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches);
void ex_panorama();

void main()
{
	//ex_panorama();
	Mat book = imread("Book1.jpg", 1);
	Mat book2 = imread("Book2.jpg", 1);
	Mat book3 = imread("Book3.jpg", 1);
	Mat scene = imread("Scene.jpg", 1);
	//imshow("book1",book);
	/*findbook(book, scene , 10 , 50);
	findbook(book2, scene, 10, 50);
	findbook(book3, scene, 10, 50);*/
	//ex_panorama();
	//ex_panorama_simple();
	ex_panorama();
	waitKey(0);
}
Mat findbook(Mat img_l, Mat img_r, int thresh_dist, int min_matches)
{
	Mat img_l_re; //size�� �ʹ� Ŀ�� 0.7��
	//imshow("origin r", img_r);
	resize(img_l, img_l_re, Size(img_l.cols * 0.7, img_l.rows * 0.7));
	Mat img_r_re;//size�� �ʹ� Ŀ�� 0.7��
	resize(img_r, img_r_re, Size(img_r.cols * 0.7, img_r.rows * 0.7));
	//imshow("re scene", img_r_re);
	//<Gray scale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l_re, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r_re, img_gray_r, CV_BGR2GRAY);

	//<Ư¡��<key points> ���� // SiFT���
	//Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	Ptr <cv::SiftFeatureDetector> Detector = SiftFeatureDetector::create();
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);//0.7��� �̹���
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r; // 0.7��� �̹������� SIFT�� ����ؼ� Ư¡�� �̾Ƴ� ���
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_r.png", img_kpts_r);

	//<����� (descriptor) ����> SIFT���
	Ptr <cv::SiftDescriptorExtractor> detector = SiftDescriptorExtractor::create();

	Mat img_des_obj, img_des_scene;
	detector->compute(img_gray_l, kpts_obj, img_des_obj);
	detector->compute(img_gray_r, kpts_scene, img_des_scene);

	//<����ڸ� �̿��� Ư¡�� ��Ī> brute force ��Ī ���
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_r, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);
	imshow("matches", img_matches);
	//<��Ī ��� ����>
	//��Ī �Ÿ��� ���� ����� ��Ī �ܷΰ��� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60 �̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++)
	{
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;

	}
	printf("max_dist : %f \n", dist_max);
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good; // DMatch��ü�� ���Ϳ��� threshold�� �̿��� ��Ī����� ���͸� �Ѵ�. 
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++)
		{
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);

		}
		matches_good = good_matches2;
		thresh_dist -= 1;

	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);
	Mat img_matches_good_re;
	resize(img_matches_good, img_matches_good_re, Size(img_matches_good.cols*0.7, img_matches_good.rows*0.7));
	imshow("match good", img_matches_good);//����
	imshow("result��", img_matches_good_re);//0.7�� ���
	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene; // ��ǥ�� Point2f��ü ���� �ȿ� �����Ѵ�.
	for (int i = 0; i < matches_good.size(); i++)
	{
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); // img2

	}

	//<��Ī ����κ��� homography ����� ����
	Mat mat_homo = findHomography(obj, scene, RANSAC); // homography��ȯ ���
	
													   //�̻�ġ ���Ÿ� ���� RANSAC�߰�
	//<Homograpy ����� �̿��� ���� ����ȯ>
	//!!!!!!!!!!!!!!!!!!!!!!homography��Ŀ� ������ �� �ִ� ���
	//Mat img_result;//å�� ��Ŀ� ���� ��� ��ȯ�Ǵ��� Ȯ���ϴ� ����
	//Mat img_l_rect = img_l.clone(); // �׵θ� å�� ��� ��ȯ�Ǵ� �� Ȯ�� 
	//rectangle(img_l_rect, Rect(0, 0, img_l.cols, img_l.rows),Scalar(255.255,255) , 50, 4,0);
	////imshow("rec line", img_l_rect);
	//warpPerspective(img_l_rect, img_result, mat_homo, Size(img_l.cols*2 , img_l.rows*2 ), INTER_CUBIC);
	////������ �߸��� ���� �����ϱ� ���� �������� �ο� 
	////imshow("result", img_result);
	//Mat img_result_re;
	//
	//resize(img_result,img_result_re ,Size(img_result.cols*0.7, img_result.rows*0.7));
	//imshow("result", img_result_re);//warp perspective �� �� ���� �� 
	//<���� ����� ����ȯ�� ���� ���� ��ü>
	Mat img_book; // å ���� �ش��ϴ� �κ�
	img_book = img_matches_good.clone(); // matching�� ���͸� �� ������ ����
	
	Mat roi(img_book, Rect(0, 0, img_l_re.cols, img_l_re.rows)); // matching�� ���͸� �� ���� �� ã������ �ϴ� å�� ���� ���� ����
	//imshow("result 2", img_result_re);
	//rectangle(roi, Rect(0, 0, img_l.cols, img_l.rows), Scalar(0, 255, 0), 50, 4, 0);
	//img_matches_good_re.copyTo(roi);
	imshow("roi", roi);//å�� �ش�
	//imshow("pano", img_pano);
	
	//<Point2f��ü ���� ����> ��ǥ ��ü�� ���� perspectiveTransform ���� ���༱�� �����Ǳ� ������ 
	vector<Point2f> srcQuad(4);
	srcQuad[0] = Point2f(0,0);
	srcQuad[1] = Point2f(roi.cols,0);
	srcQuad[2] = Point2f(roi.cols,roi.rows);
	srcQuad[3] = Point2f(0,roi.rows);

	vector<Point2f> dstQuad(4);
	perspectiveTransform(srcQuad,dstQuad, mat_homo);
	Point2f p(img_l_re.cols, 0);
	line(img_matches_good, dstQuad[0]+p, dstQuad[1]+p, Scalar(255.255,255) ,10 ,8,0  );
	line(img_matches_good, dstQuad[1] + p, dstQuad[2] + p, Scalar(255.255, 255), 10, 8, 0);
	line(img_matches_good, dstQuad[2] + p, dstQuad[3] + p, Scalar(255.255, 255), 10, 8, 0);
	line(img_matches_good, dstQuad[3] + p, dstQuad[0] + p, Scalar(255.255, 255), 10, 8, 0);
	/*line(img_matches_good_re, dstQuad[1], dstQuad[2], Scalar(0.255, 0));
	line(img_matches_good_re, dstQuad[2], dstQuad[3], Scalar(0.255, 0));
	line(img_matches_good_re, dstQuad[3], dstQuad[4], Scalar(0.255, 0));*/
	imshow("final", img_matches_good);
	waitKey(0);
	destroyAllWindows();

	return img_matches_good;
}
void ex_panorama_simple()
{
	Mat img;
	vector<Mat> imgs;

	img = imread("expanoleft.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("expanocenter.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	img = imread("expanoright.jpg", IMREAD_COLOR);
	imgs.push_back(img);
	Mat result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false);
	Stitcher::Status status = stitcher->stitch(imgs, result);

	if (status != Stitcher::OK)
	{
		cout << "can't stitch images, error code = " << int(status) << endl;
		exit(-1);
	}

	imshow("ex_simple_result2", result);
	imwrite("ex_simple_result2.png", result);
	waitKey();
}

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches)
{
	//<Gray scale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<Ư¡��<key points> ����
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("img_kpts_lnew.png", img_kpts_l);
	imwrite("img_kpts_rnew.png", img_kpts_r);

	//<����� (descriptor) ����>
	Ptr<SurfDescriptorExtractor>Extractor = SURF::create(100, 4, 3, false, true);

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);
	cout << "������� ����" << endl;
	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_r, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_new.png", img_matches);
	cout << "��Ī���� ��" << endl;
	//<��Ī ��� ����>
	//��Ī �Ÿ��� ���� ����� ��Ī �ܷΰ��� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60 �̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++)
	{
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;

	}
	printf("max_dist : %f \n", dist_max);
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++)
		{
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);

		}
		matches_good = good_matches2;
		thresh_dist -= 1;

	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good_new.png", img_matches_good);
	cout << "���� ��� ���� ����" << endl;
	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++)
	{
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); // img2

	}

	//<��Ī ����κ��� homography ����� ����
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	//�̻�ġ ���Ÿ� ���� RANSAC�߰�

	//<Homograpy ����� �̿��� ���� ����ȯ>
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//������ �߸��� ���� �����ϱ� ���� �������� �ο� 

	//<���� ����� ����ȯ�� ���� ���� ��ü>
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);

	//<���� ���� �߶󳻱�>
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++)
	{
		for (int x = 0; x < img_pano.cols; x++)
		{

			if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
				img_pano.at<Vec3b>(y, x)[1] == 0 && img_pano.at<Vec3b>(y, x)[2] == 0)
			{
				continue;
			}
			if (cut_x < x) cut_x = x;
			if (cut_y < y) cut_y = y;
		}

	}
	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
	imwrite("img_pano_cut_new.png", img_pano_cut);

	return img_pano_cut;
}

void ex_panorama()
{
	Mat matImage1 = imread("expanocenter.jpg", IMREAD_COLOR);
	Mat matImage2 = imread("expanoleft.jpg", IMREAD_COLOR);
	Mat matImage3 = imread("expanoright.jpg", IMREAD_COLOR);
	if (matImage1.empty() || matImage2.empty() || matImage3.empty()) exit(-1);

	Mat result;
	flip(matImage1, matImage1, 1);
	flip(matImage2, matImage2, 1);
	result = makePanorama(matImage1, matImage2, 3, 60);
	flip(result, result, 1);
	result = makePanorama(result, matImage3, 3, 60);

	imshow("ex_panorama_Result", result);
	imwrite("ex_panorama_result.png", result);
	waitKey();
}

