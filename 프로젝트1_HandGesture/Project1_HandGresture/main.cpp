#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>   
#include <string>
#include <Windows.h>  // SetCursorPos()�� ���� ���� 
#pragma comment(lib, "user32.lib") //setcursorpos ���� ��� ���� �߱� 

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

		//�Ǻλ��� ������ Cb,Cr�� ����: Cb: 77 ~ 127 / Cr: 133 ~ 173

		//���� ������ YCrCB�� �����Ѵ�.
		cvtColor(tmpImg, handImg, COLOR_BGR2YCrCb);
		inRange(handImg, Scalar(0, 133, 77), Scalar(255, 173, 127), handImg);
		//�������� ó������ ���� ����ϰ� �и�
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

		//������ �׸��� -> yellow
		drawContours(tmpImg, contours, -1, Scalar(0, 255, 255), 1, 8, vector < Vec4i>(), 0, Point());

		//�չٴ� �߽��� �׸���!
		Point center = getHandCenter(mask, radius);
		circle(tmpImg, center, 2, Scalar(0, 255, 0), -1);

		//�հ��� ����
		int fingerNum = 0;
		fingerNum = getFingerCount(mask, center, radius, 0);
		// ���� puttext �غ��� 

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

	// ���׸��� 
	Mat cImg(mask.size(), CV_8U, Scalar(0));
	circle(cImg, center, radius * scale, Scalar(255));

	//���� �ܰ����� ������ ����
	vector<vector<Point>> contours;
	findContours(cImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (contours.size() == 0)  //�ܰ����� ������, �� ���� ������
		return -1;

	int fingerCount = 0;
	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1)
			fingerCount++;
	}

	//�ո�� ������ �� ���� 1���� �����Ѵ�.
	return fingerCount - 1;
}

Point getHandCenter(Mat& mask, double& radius) {
	Mat dst;
	distanceTransform(mask, dst, DIST_L2, 5);

	int maxIdx[2];
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);
	return Point(maxIdx[1], maxIdx[0]);
}