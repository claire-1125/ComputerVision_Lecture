#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// 차선 검출용 각도 범위
int r_min_val = 10;
int r_max_val = 70;
int l_min_val = 120;
int l_max_val = 180;

int min_len_val = 10;
int y_tresh_val = 60;
double PI = 3.14;

// trackbar 위의 포인트 옮기기 위해 동적 설정 
int heightLeftValue = 4;
int heightRightValue = 4;
int heightMax = 6;

Mat ROI(Mat image, Point* pt) {

	Mat img_mask = Mat::zeros(image.rows, image.cols, CV_8UC1);

	Scalar ignore_mask_color = Scalar(255, 255, 255);
	const Point* ppt[1] = { pt };
	int numpt[] = { 4 };

	//채워진 다각형 그리기 pts
	fillPoly(img_mask, ppt, numpt, 1, Scalar(255, 255, 255), LINE_8);

	Mat img_masked;
	//image 와 img_mask의 비트 and 연산 결과 를 img_masked로 도출
	//pixel 값이 0 이 아닌 부부만 도출하게 된다. 
	bitwise_and(image, img_mask, img_masked);  // roi 영역 나머지는 검정색이 되버린다. 

	return img_masked;
}

Point Intersection(const Point* p1, const Point* p2, const Point* p3, const Point* p4) {

	Point result;

	result.x = ((p1->x * p2->y - p1->y * p2->x) * (p3->x - p4->x) - (p1->x - p2->x) * (p3->x * p4->y - p3->y * p4->x)) / ((p1->x - p2->x) * (p3->y - p4->y) - (p1->y - p2->y) * (p3->x - p4->x));

	result.y = ((p1->x * p2->y - p1->y * p2->x) * (p3->y - p4->y) - (p1->y - p2->y) * (p3->x * p4->y - p3->y * p4->x)) / ((p1->x - p2->x) * (p3->y - p4->y) - (p1->y - p2->y) * (p3->x - p4->x));

	return result;
}

void colorSelect(Mat image, Mat& color_selected) {
	Mat img;
	image.copyTo(img);
	Mat imgCom;
	Mat white_mask, wImg;
	Mat blue_mask, bImg;
	// scalar(색상, 채도, 명도)
	Scalar lower_white = Scalar(10, 40, 0); //흰색 차선 (HSL)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_blue = Scalar(230, 36, 38);
	Scalar upper_blue = Scalar(255, 255, 255);

	cvtColor(img, img, COLOR_RGB2HLS);

	// white pixel
	inRange(img, lower_white, upper_white, white_mask);
	bitwise_and(img, img, wImg, white_mask);
	//blue
	inRange(img, lower_blue, upper_blue, blue_mask);
	bitwise_and(img, img, bImg, blue_mask);

	addWeighted(wImg, 1.0, bImg, 1.0, 0.0, imgCom);
	imgCom.copyTo(color_selected);
}

class Histogram1D {
private:
	int histSize[1]; // 히스토그램 빈도수
	float hranges[2]; // 히스토그램 최소/최대 화소값
	const float* ranges[1];
	int channels[1]; // 1채널만 사용
public:
	Histogram1D() { // 1차원 히스토그램을 위한 인자 준비
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0;
	}
	// 1차원 히스토그램 계산
	MatND getHistogram(const Mat& image) {
		MatND hist;
		// 이미지의 히스토그램 계산
		calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
		return hist;
	}
	MatND equalize(const cv::Mat& image) {
		cv::Mat result;
		cv::equalizeHist(image, result);
		return result;
	}
};


int main(int argc, char** argv) {

	VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		cout << "no video available" << endl;
		return -1;
	}
	bool playVideo = true;
	Mat edges, frame, gframe, blframe, chFrame;

	namedWindow("roi_edges");
	namedWindow("original");

	for (;;) {
		if (playVideo)
		{
			cap >> frame;
			resize(frame, chFrame, Size(700, 700), 0, 0, 1);

			// 파랑, 흰색 범위 내에 있는 부분만 차선후보로 따로 저장함 
			Mat color_selected;
			colorSelect(chFrame, color_selected);
			cvtColor(color_selected, gframe, COLOR_BGR2GRAY);

			gframe = gframe + 160;

			//평활화 수행
			Histogram1D h;
			MatND eqHist = h.equalize(gframe);
			
			GaussianBlur(eqHist, blframe, Size(7, 7), 1.5, 1.5); //원래 size(7,7)

			Canny(blframe, edges, 50, 150);

			int width = chFrame.cols;
			int height = chFrame.rows;

			double bottomLeftValue = 0.92;
			double bottomRightValue = 0.92;

			//trackbar
			createTrackbar("right point", "original", &heightRightValue, heightMax);
			createTrackbar("left point", "original", &heightLeftValue, heightMax);


			Point points[4];
			points[0] = Point((width * (1 - 0.85)) / 2, height * 0.92);					               // 왼쪽 아래
			points[1] = Point(width - (width * 1.2) / 2, height - height * heightLeftValue / 10);		   // 왼쪽 위
			points[2] = Point(width - (width * 0.93) / 2, height - height * heightRightValue / 10);       // 오른쪽 위
			points[3] = Point(width - (width * 0.15) / 2, height * 0.92);			                  	//오른쪽 아래 

			//ROI
			Mat roi_edges;
			roi_edges = ROI(edges, points);

			//roi_edge영역에서의 직선만 구하게 되어 다른 불순물 검출이 줄어든다. 
			vector<Vec2f> lne;

			HoughLines(roi_edges, lne, 1, PI / 180, 130);
			Mat result(edges.rows, edges.cols, CV_8U, cv::Scalar(255));
			vector<Vec2f>::const_iterator it = lne.begin();
			while (it != lne.end()) {
				float rho = (*it)[0]; // 첫 번째 요소는 rho 거리
				float theta = (*it)[1]; // 두 번째 요소는 델타 각도
				if (theta < PI / 4. || theta > 3. * PI / 4.)// 수직에 해당하는			
				{
					Point pt1(rho / cos(theta), 0); // 첫 행에서 해당 선의 교차점
					Point pt2((rho - edges.rows * sin(theta)) / cos(theta), edges.rows);
					// 마지막 행에서 해당 선의 교차점
					line(roi_edges, pt1, pt2, Scalar(255), 1, 8); // 하얀 선으로 그리기
				}
				++it;
			}



			//pt
			Point inter_upper, inter_lower;

			// 선 감지 위한 허프 변환
			vector<Vec4i> lines;
			size_t i;
			HoughLinesP(roi_edges, lines, 1, CV_PI / 180, 50, /*min_len_val = 10*/ 70, 40);

			for (i = 0; i < lines.size(); i++) {
				Vec4i l = lines[i];
				double theta = atan2(l[2] - l[0], l[3] - l[1]) * 180 / CV_PI;
				if ((r_min_val <= theta && theta <= r_max_val) || (l_min_val <= theta && theta <= l_max_val)) {
					if ((l[3] > y_tresh_val * width / 100) && (l[1] > y_tresh_val * height / 100)) {

						// roi의 위의 가로선과 houghline 직선이 만나는 교점과, roi의 아래 가로선과 houghline직선이 만나는 교점을 구한다.
						inter_upper = Intersection(&points[1], &points[2], &Point(l[0], l[1]), &Point(l[2], l[3]));
						inter_lower = Intersection(&points[0], &points[3], &Point(l[0], l[1]), &Point(l[2], l[3]));
						//확인차 그려봄
						circle(chFrame, inter_upper, 4, Scalar(0, 255, 255), 1, 8);
						circle(chFrame, inter_lower, 4, Scalar(0, 0, 255), 1, 8);
						//line그리기 => 교점과 교점을 이어주게 된다. 
						line(chFrame, inter_upper, inter_lower, Scalar(0, 0, 255), 2, 8);

						int s;
						s = abs(inter_upper.x - inter_lower.x);
						if (s >= 0 && s < 35) {
							putText(chFrame, "WARNING!", Point(10, 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 4, 8);
						}

					}
				}
			}

			//ROI 영역 POINT 표시해주기 
			circle(chFrame, points[0], 3, Scalar(0, 0, 0), 1, 8);
			circle(chFrame, points[1], 3, Scalar(255, 0, 0), 1, 8); //왼쪽 위
			circle(chFrame, points[2], 3, Scalar(0, 255, 0), 1, 8); //오른쪽 위
			circle(chFrame, points[3], 3, Scalar(0, 0, 255), 1, 8);

			imshow("roi_edges", roi_edges);
			imshow("original", chFrame);
		}

		int key = waitKey(30);
		if (key == 'q')  // 프로그램 종료
			break;

		else if (key == 's')  // 영상정지
			playVideo = !playVideo;
	}
	cap.release();
	return 0;
}