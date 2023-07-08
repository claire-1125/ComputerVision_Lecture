#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>   
#include <string>
#include <Windows.h>  // SetCursorPos()를 위해 포함 
#pragma comment(lib, "user32.lib") //setcursorpos 에러 잡기 위해 추기 

using namespace cv;
using namespace std;

Point getHandCenter(Mat& mask, double& radius);
int getFingerCount(const Mat& mask, Point center, double radius, double scale);


int erosion_value = 0;
int const max_erosion = 2;
int erosion_size = 0;
int const ersion_max_size = 21;
int dilation_value = 0;
int dilation_size = 0;
int erosion_type = 0;
int drawX, drawY;

int main(int argc, char** argv)
{
	Mat element;
	Mat frame, tmpImg;
	Mat handImg, mask, mask1;
	Point dst;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	double radius = 5;
	VideoCapture video(0);

	if (!video.isOpened()) {
		cout << "video not open error!" << endl;
		return 0;
	}

	namedWindow("change_image");
	namedWindow("original_image");
	createTrackbar("ele_erosion", "original_image", &erosion_value, max_erosion); //2
	createTrackbar("erosion_size", "original_image", &erosion_size, ersion_max_size); //21
	createTrackbar("ele_dilation", "original_image", &dilation_value, max_erosion);   // 2
	createTrackbar("dilation_size", "original_image", &dilation_size, ersion_max_size);  //21

	cout << "if you want to move cursor, make your finger Zero!" << endl;

	while (true)
	{
		video >> tmpImg;

		if (tmpImg.empty()) break;

		if (erosion_value == 0) erosion_type = MORPH_RECT;
		else if (erosion_value == 1) erosion_type = MORPH_CROSS;
		else if (erosion_value == 2) erosion_type = MORPH_ELLIPSE;

		element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

		//피부색이 가지는 Cb,Cr의 영역: Cb: 77 ~ 127 / Cr: 133 ~ 173

		//먼저 영상을 YCrCB로 변경한다.
		cvtColor(tmpImg, handImg, COLOR_BGR2YCrCb);
		inRange(handImg, Scalar(0, 133, 77), Scalar(255, 173, 127), handImg);
		//열림으로 처리해줘 좀더 깔끔하게 분리
		erode(handImg, handImg, element);
		dilate(handImg, handImg, element);

		mask = handImg.clone();

		findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Eliminate too short or too long contours
		int cmin = 400; // minimum contour length
		int cmax = 1000; // maximum contour length
		vector<vector <Point>> ::const_iterator itc = contours.begin();
		while (itc != contours.end() - 1) {
			if (itc->size() - 1 < cmin || itc->size() - 1 > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}

		//컨투어 그리기 -> yellow
		drawContours(tmpImg, contours, -1, Scalar(0, 255, 255), 1, 8, vector < Vec4i>(), 0, Point());

		//손바닥 중심점 그리기!
		Point center = getHandCenter(mask, radius);
		circle(tmpImg, center, 2, Scalar(0, 255, 0), -1);

		//손가락 개수
		int fingerNum = 0;
		fingerNum = getFingerCount(mask, center, radius, 0);
		// 영상에 puttext 해보자 

		string num;
		switch (fingerNum) {
		case 0: num = "zero"; break;
		case 1: num = "one"; break;
		case 2: num = "two"; break;
		case 3: num = "three"; break;
		case 4: num = "four"; break;
		case 5: num = "five"; break;
		default: break;
		}

		Point p;
		p.x = 10; p.y = 30;
		putText(tmpImg, "finger Count : " + num, p, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 4, 8);


		if (num.compare("zero") == 0) {
			SetCursorPos(center.x, center.y);
		}

		imshow("change_image", handImg);
		imshow("original_image", tmpImg);

		if (waitKey(10) == 27) {
			break;
		}
	}

	video.release();
	tmpImg.release();
	frame.release();

	destroyAllWindows();

	return 0;
}

int getFingerCount(const Mat& mask, Point center, double radius, double scale) {
	scale = 2.0;

	// 원그리기 
	Mat cImg(mask.size(), CV_8U, Scalar(0));
	circle(cImg, center, radius * scale, Scalar(255));

	//원의 외곽선을 저장할 벡터
	vector<vector<Point>> contours;
	findContours(cImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (contours.size() == 0)  //외곽선이 없을때, 즉 손이 없을때
		return -1;

	int fingerCount = 0;
	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1)
			fingerCount++;
	}

	//손목과 만나는 점 개수 1개는 제외한다.
	return fingerCount - 1;
}

Point getHandCenter(Mat& mask, double& radius) {
	Mat dst;
	distanceTransform(mask, dst, DIST_L2, 5);

	int maxIdx[2];
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);
	return Point(maxIdx[1], maxIdx[0]);
}