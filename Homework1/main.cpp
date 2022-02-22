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
	putTextZH(src, "胡成成-2101210578", Point(200, 600), Scalar(255, 0, 0), 100, "楷体");


	imshow("src", src);

	imwrite("test_opencv.jpg", src);

	waitKey(0);//延时30毫秒

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}

	//循环显示每一帧
	while (1)
	{
		Mat frame;//存储每一帧图像
		cap >> frame;//读取当前帧
		putTextZH(frame, "胡成成-2101210578", Point(100, 200), Scalar(0, 255, 0), 30, "微软雅黑");

		putText(frame, "Huchengcheng-2101210578", Point(100, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 0));
		imshow("读取视频", frame);
		waitKey(30);//延时30毫秒
	}
	return 0;
}

