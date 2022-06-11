#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main() {
	 
	Mat src = imread("D:/OpenCV/picture zone/shebaoka.png");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}

	//图像前期处理
	Mat src_gray;
	filter2D(src,src_gray,-1,2);
	imshow("2D",src_gray);
	cvtColor(src_gray,src_gray,COLOR_BGR2GRAY);
	equalizeHist(src_gray, src_gray);
	imshow("src_gray", src_gray);

	//二值化
	Mat src_binary;
	threshold(src_gray,src_binary,0,255,THRESH_BINARY_INV|THRESH_OTSU);
	imshow("binary",src_binary);

	//形态学处理
	Mat src_close;
	Mat kernel = getStructuringElement(MORPH_RECT,Size(5,5),Point(-1,-1));
	morphologyEx(src_binary,src_close,MORPH_CLOSE,kernel,Point(-1,-1),3);
	imshow("close",src_close);

	//寻找轮廓
	bitwise_not(src_close,src_close,Mat());
	vector<vector<Point>> controus;
	vector<Vec4i>hireachy;
	findContours(src_close,controus,hireachy,CV_RETR_TREE,CHAIN_APPROX_SIMPLE,Point());

	
	//绘制轮廓
	int wdith = src.cols;
	int hight = src.rows;
	
	Mat src_drawcontrous = Mat::zeros(src.size(),CV_8UC3);
	for (size_t i = 0; i < controus.size(); i++)
	{
		//查找最小外接矩形
		Rect rect = boundingRect(controus[i]);
		if (rect.width>(wdith/2) && rect.width<(wdith-5))
		{
			drawContours(src_drawcontrous,controus,static_cast<int>(i),Scalar(0,0,255),2,8,hireachy,0,Point());
		}
	}
	imshow("outLine",src_drawcontrous);
	

	//霍夫提取4条线
	vector<Vec4i> hough_line;
	Mat contoursImg;
	int accu = min(wdith*0.5,hight*0.5);
	cvtColor(src_drawcontrous, contoursImg,COLOR_BGR2GRAY);
	HoughLinesP(contoursImg,hough_line,1,CV_PI/180.0,accu,accu,0);
	Mat houghLineImg = Mat::zeros(src.size(),CV_8UC3);
	for (size_t i = 0; i < hough_line.size(); i++)
	{
		Vec4i ln = hough_line[i];
		line(houghLineImg, Point(ln[0], ln[1]), Point(ln[2],ln[3]),Scalar(0,0,255),2,8,0);
	
	}
	printf("number of line:%d\n", hough_line.size());
	imshow("hough_line",houghLineImg); 


	//寻找与定位上下左右4条线
	int deltah = 0;
	Vec4i topLine, bottomLine;
	Vec4i leftLine, rightLine;
	for (size_t i = 0; i < hough_line.size(); i++)
	{
		Vec4i ln = hough_line[i];
		deltah = abs(ln[3]- ln[1]);
		if (ln[3] < hight / 2.0 && ln[1] < hight / 2.0 && deltah < accu - 1 )
		{
			topLine = hough_line[i];
		}
		if (ln[3] > hight / 2.0 && ln[1] > hight / 2.0 && deltah < accu - 1)
		{
			bottomLine = hough_line[i];
		}
		if (ln[0] < wdith / 2.0 && ln[2] < wdith / 2.0)
		{
			leftLine = hough_line[i];
		}
		if (ln[0] > wdith / 2.0 && ln[2] > wdith / 2.0)
		{
			rightLine = hough_line[i];
		}
	}
	cout << "topline:p1(x,y)=" << topLine[0] << "," << topLine[1] << "P2(x,y)="<<topLine[2] << "," << topLine[3] << endl;
	cout << "bottomLine:p1(x,y)=" << bottomLine[0] << "," << bottomLine[1] << "P2(x,y)=" << bottomLine[2] << "," << bottomLine[3] << endl;
	cout << "leftline:p1(x,y)=" << leftLine[0] << "," << leftLine[1] << "P2(x,y)=" << leftLine[2] << "," << leftLine[3] << endl;
	cout << "rightline:p1(x,y)=" << rightLine[0] << "," << rightLine[1] << "P2(x,y)=" << rightLine[2] << "," << rightLine[3] << endl;
	
	//拟合四条直线方程
		//求每条线的方程，用于转换   y=kx+c
	float k1, c1;
		//线段的斜率 = 两点坐标（x-x）/（y-y）
	k1 = float(topLine[3]-topLine[1])/float(topLine[2]-topLine[0]); 
		//c = y-kx  任意点坐标都可出c
	c1 = topLine[1] - k1 * topLine[0];

	float k2, c2;
	k2 = float(bottomLine[3] - bottomLine[1]) / float(bottomLine[2] - bottomLine[0]);
	c2 = bottomLine[1] - k2 * bottomLine[0];
	
	float k3, c3;
	k3 = float(leftLine[3] - leftLine[1]) / float(leftLine[2] - leftLine[0]);
	c3 = leftLine[1] - k3 * leftLine[0];

	float k4, c4;
	k4 = float(rightLine[3] - rightLine[1]) / float(rightLine[2] - rightLine[0]);
	c4 = rightLine[1] - k4 * rightLine[0];


	// 四条直线交点
	Point p1; // 左上角
	p1.x = static_cast<int>((c1 - c3) / (k3 - k1));
	p1.y = static_cast<int>(k1 * p1.x + c1);
	Point p2; // 右上角
	p2.x = static_cast<int>((c1 - c4) / (k4 - k1));
	p2.y = static_cast<int>(k1 * p2.x + c1);
	Point p3; // 左下角
	p3.x = static_cast<int>((c2 - c3) / (k3 - k2));
	p3.y = static_cast<int>(k2 * p3.x + c2);
	Point p4; // 右下角
	p4.x = static_cast<int>((c2 - c4) / (k4 - k2));
	p4.y = static_cast<int>(k2 * p4.x + c2);
	cout << "p1(x, y)=" << p1.x << "," << p1.y << endl;
	cout << "p2(x, y)=" << p2.x << "," << p2.y << endl;
	cout << "p3(x, y)=" << p3.x << "," << p3.y << endl;
	cout << "p4(x, y)=" << p4.x << "," << p4.y << endl;

	
	// 显示四个点坐标
	circle(houghLineImg, p1, 2, Scalar(0, 0, 255), 2, 8, 0);
	circle(houghLineImg, p2, 2, Scalar(0, 0, 255), 2, 8, 0);
	circle(houghLineImg, p3, 2, Scalar(0, 0, 255), 2, 8, 0);
	circle(houghLineImg, p4, 2, Scalar(0, 0, 255), 2, 8, 0);
	line(houghLineImg, Point(topLine[0], topLine[1]), Point(topLine[2], topLine[3]), Scalar(0, 255, 0), 2, 8, 0);
	imshow("four corners", houghLineImg);
	

	//透视变换
		//拿到4个点
	vector<Point2f>  src_corners(4);
	src_corners[0] = p1;
	src_corners[1] = p2;
	src_corners[2] = p3;
	src_corners[3] = p4;

	vector<Point2f> dst_corners(4);
	dst_corners[0] = Point(0, 0);
	dst_corners[1] = Point(wdith, 0);
	dst_corners[2] = Point(0, hight);
	dst_corners[3] = Point(wdith, hight);

	// 获取透视变换矩阵
	Mat resultImage;
	Mat warpmatrix = getPerspectiveTransform(src_corners, dst_corners);
	warpPerspective(src, resultImage, warpmatrix, resultImage.size(), INTER_LINEAR);
	namedWindow("Final Result", CV_WINDOW_AUTOSIZE);
	imshow("Final Result", resultImage);


	waitKey(0);
	return 0;
}