#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include "putText.h"

using namespace std;
using namespace cv;

int main() {
	
	Mat src = imread("test.jpg");

	putText(src, "Huchengcheng-2101210578", Point(200, 200), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(0, 0, 255));
	putTextZH(src, "���ɳ�-2101210578", Point(200, 600), Scalar(255, 0, 0), 100, "����");


	imshow("src", src);

	imwrite("test_opencv.jpg", src);

	waitKey(0);//��ʱ30����

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}

	//ѭ����ʾÿһ֡
	while (1)
	{
		Mat frame;//�洢ÿһ֡ͼ��
		cap >> frame;//��ȡ��ǰ֡
		putTextZH(frame, "���ɳ�-2101210578", Point(100, 200), Scalar(0, 255, 0), 30, "΢���ź�");

		putText(frame, "Huchengcheng-2101210578", Point(100, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 0));
		imshow("��ȡ��Ƶ", frame);
		waitKey(30);//��ʱ30����
	}
	return 0;
}

