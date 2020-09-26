#pragma once
// ֻ�ʺ�3.4.2�汾�����ϵ�opencv
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
	//��������  init
	//���ܣ�   �㷨��ʼ������һ��
	//���������modelConfiguration             darknet��ܵ������ļ���cfg��׺
	//        modelBinary                    darknet��ܵ�ģ�Ͳ����ļ���weights��׺
	//        classesFile                    ����ļ�
	//        confThreshold                  ���Ŷ���ֵ
	//        nmsThreshold                   �Ǽ���ֵ������ֵ
	//�����������
	//����ֵ�� ��
	//�޸ļ�¼��
	//cuixingxing          2018-09-05      ����
	********************************************************************/
	void init(const String& modelConfiguration,const String& modelBinary,const string & classesFile, float confThreshold=0.1, float nmsThreshold = 0.1);

	/*****************************************************************
	//��������  detect
	//���ܣ�   �Ե���ͼ�����Ŀ����
	//���������image             Mat����
	//���������predictROIs       Ԥ��Ŀ���ROI
	//        predictScores      Ԥ��ķ���
	//        predictLabels      Ԥ��ı�ǩ
	//����ֵ�� ��       
	//�޸ļ�¼��
	//cuixingxing          2018-09-05      ����
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
		vector<Rect> &predictROIs, vector<float> &predictScores, vector<string> &predictLabels); // ���ROI������,���
};



