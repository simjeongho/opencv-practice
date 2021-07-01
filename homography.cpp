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
	Mat img_l_re; //size가 너무 커서 0.7배
	//imshow("origin r", img_r);
	resize(img_l, img_l_re, Size(img_l.cols * 0.7, img_l.rows * 0.7));
	Mat img_r_re;//size가 너무 커서 0.7배
	resize(img_r, img_r_re, Size(img_r.cols * 0.7, img_r.rows * 0.7));
	//imshow("re scene", img_r_re);
	//<Gray scale로 변환>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l_re, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r_re, img_gray_r, CV_BGR2GRAY);

	//<특징점<key points> 추출 // SiFT사용
	//Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	Ptr <cv::SiftFeatureDetector> Detector = SiftFeatureDetector::create();
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);//0.7배된 이미지
	Detector->detect(img_gray_r, kpts_scene);

	//<특징점 시각화>
	Mat img_kpts_l, img_kpts_r; // 0.7배된 이미지에서 SIFT를 사용해서 특징점 뽑아낸 행렬
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_r.png", img_kpts_r);

	//<기술자 (descriptor) 추출> SIFT사용
	Ptr <cv::SiftDescriptorExtractor> detector = SiftDescriptorExtractor::create();

	Mat img_des_obj, img_des_scene;
	detector->compute(img_gray_l, kpts_obj, img_des_obj);
	detector->compute(img_gray_r, kpts_scene, img_des_scene);

	//<기술자를 이용한 특징점 매칭> brute force 매칭 사용
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<매칭 결과 시각화>
	Mat img_matches;
	drawMatches(img_gray_r, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);
	imshow("matches", img_matches);
	//<매칭 결과 정제>
	//매칭 거리가 작은 우수한 매칭 겨로가를 정제하는 과정
	//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60 이상 까지 정제
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

	vector<DMatch> matches_good; // DMatch객체의 벡터에서 threshold를 이용해 매칭결과를 필터링 한다. 
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

	//<우수한 매칭 결과 시각화>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);
	Mat img_matches_good_re;
	resize(img_matches_good, img_matches_good_re, Size(img_matches_good.cols*0.7, img_matches_good.rows*0.7));
	imshow("match good", img_matches_good);//원본
	imshow("result다", img_matches_good_re);//0.7배 결과
	//<매칭 결과 좌표 추출>
	vector<Point2f> obj, scene; // 좌표를 Point2f객체 벡터 안에 저장한다.
	for (int i = 0; i < matches_good.size(); i++)
	{
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); // img2

	}

	//<매칭 결과로부터 homography 행렬을 추출
	Mat mat_homo = findHomography(obj, scene, RANSAC); // homography변환 행렬
	
													   //이상치 제거를 위해 RANSAC추가
	//<Homograpy 행렬을 이용해 시점 역변환>
	//!!!!!!!!!!!!!!!!!!!!!!homography행렬에 접근할 수 있는 방법
	//Mat img_result;//책이 행렬에 의해 어떻게 변환되는지 확인하는 영상
	//Mat img_l_rect = img_l.clone(); // 테두리 책이 어떻게 변환되는 지 확인 
	//rectangle(img_l_rect, Rect(0, 0, img_l.cols, img_l.rows),Scalar(255.255,255) , 50, 4,0);
	////imshow("rec line", img_l_rect);
	//warpPerspective(img_l_rect, img_result, mat_homo, Size(img_l.cols*2 , img_l.rows*2 ), INTER_CUBIC);
	////영상이 잘리는 것을 방지하기 위해 여유공간 부여 
	////imshow("result", img_result);
	//Mat img_result_re;
	//
	//resize(img_result,img_result_re ,Size(img_result.cols*0.7, img_result.rows*0.7));
	//imshow("result", img_result_re);//warp perspective 한 후 줄인 거 
	//<기준 영상과 역변환된 시점 영상 합체>
	Mat img_book; // 책 영상만 해당하는 부분
	img_book = img_matches_good.clone(); // matching이 필터링 된 영상을 복제
	
	Mat roi(img_book, Rect(0, 0, img_l_re.cols, img_l_re.rows)); // matching이 필터링 된 영역 중 찾으려고 하는 책에 대한 관심 영역
	//imshow("result 2", img_result_re);
	//rectangle(roi, Rect(0, 0, img_l.cols, img_l.rows), Scalar(0, 255, 0), 50, 4, 0);
	//img_matches_good_re.copyTo(roi);
	imshow("roi", roi);//책만 해당
	//imshow("pano", img_pano);
	
	//<Point2f객체 벡터 생성> 좌표 객체를 통해 perspectiveTransform 수행 평행선이 보존되기 때문에 
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
	//<Gray scale로 변환>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<특징점<key points> 추출
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<특징점 시각화>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("img_kpts_lnew.png", img_kpts_l);
	imwrite("img_kpts_rnew.png", img_kpts_r);

	//<기술자 (descriptor) 추출>
	Ptr<SurfDescriptorExtractor>Extractor = SURF::create(100, 4, 3, false, true);

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);
	cout << "여기까지 했음" << endl;
	//<기술자를 이용한 특징점 매칭>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<매칭 결과 시각화>
	Mat img_matches;
	drawMatches(img_gray_r, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_new.png", img_matches);
	cout << "매칭까진 함" << endl;
	//<매칭 결과 정제>
	//매칭 거리가 작은 우수한 매칭 겨로가를 정제하는 과정
	//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60 이상 까지 정제
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

	//<우수한 매칭 결과 시각화>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good_new.png", img_matches_good);
	cout << "좋은 결과 까진 뽑음" << endl;
	//<매칭 결과 좌표 추출>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++)
	{
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); // img2

	}

	//<매칭 결과로부터 homography 행렬을 추출
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	//이상치 제거를 위해 RANSAC추가

	//<Homograpy 행렬을 이용해 시점 역변환>
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//영상이 잘리는 것을 방지하기 위해 여유공각 부여 

	//<기준 영상과 역변환된 시점 영상 합체>
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);

	//<검은 여백 잘라내기>
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

