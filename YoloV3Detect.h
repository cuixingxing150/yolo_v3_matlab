#pragma once
// 只适合3.4.2版本及以上的opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define NET_INPUT_WIDTH 416
#define NET_INPUT_HEIGHT 416

class YoloV3Detect
{
public:
	/*****************************************************************
	//函数名：  init
	//功能：   算法初始化，仅一次
	//输入参数：modelConfiguration             darknet框架的配置文件，cfg后缀
	//        modelBinary                    darknet框架的模型参数文件，weights后缀
	//        classesFile                    类别文件
	//        confThreshold                  置信度阈值
	//        nmsThreshold                   非极大值抑制阈值
	//输出参数：无
	//返回值： 无
	//修改记录：
	//cuixingxing          2018-09-05      创建
	********************************************************************/
	void init(const String& modelConfiguration,const String& modelBinary,const string & classesFile, float confThreshold=0.1, float nmsThreshold = 0.1);

	/*****************************************************************
	//函数名：  detect
	//功能：   对单张图像进行目标检测
	//输入参数：image             Mat类型
	//输出参数：predictROIs       预测目标的ROI
	//        predictScores      预测的分数
	//        predictLabels      预测的标签
	//返回值： 无       
	//修改记录：
	//cuixingxing          2018-09-05      创建
	********************************************************************/
	void detect(const Mat & image, vector<Rect> &predictROIs, vector<float>& predictScores, vector<string>& predictLabels);

	~YoloV3Detect();

private:
	float m_confThreshold; // Confidence threshold
	float m_nmsThreshold; // Non-maximum suppression threshold
	vector<string> m_classes;
	dnn::Net m_net;

	vector<String> getOutputsNames(const Net& net);
	void  postprocess(const Mat& frame, const vector<Mat>& outs,  
		vector<Rect> &predictROIs, vector<float> &predictScores, vector<string> &predictLabels); // 输出ROI，分数,类别
};



