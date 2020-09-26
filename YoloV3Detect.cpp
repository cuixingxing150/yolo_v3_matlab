#include "YoloV3Detect.h"

void YoloV3Detect::init(const String& modelConfiguration, const String& modelBinary, const string & classesFile,
	float confThreshold,  // Confidence threshold
	float nmsThreshold) // Non-maximum suppression threshold
{
	m_confThreshold = confThreshold;
	m_nmsThreshold = nmsThreshold;
	m_net = dnn::readNetFromDarknet(modelConfiguration, modelBinary);
	//! [Initialize network]
	m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
	m_net.setPreferableTarget(DNN_TARGET_CPU);

	// Load names of classes
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) 
		m_classes.push_back(line);
	ifs.close();
}

// 
void YoloV3Detect::detect(const Mat & image,vector<Rect> &predictROIs,vector<float>& predictScores,vector<string>& predictLabels)
{
	Mat blob;
	blobFromImage(image, blob, 1 / 255.0, cvSize(NET_INPUT_WIDTH, NET_INPUT_HEIGHT), Scalar(0, 0, 0), true, false);
	m_net.setInput(blob);
	vector<Mat> outs;
	m_net.forward(outs, getOutputsNames(m_net));
	postprocess(image, outs, predictROIs, predictScores, predictLabels);
}


vector<String> YoloV3Detect::getOutputsNames(const Net& net)
{
	vector<String> names;
	if (names.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers();
		vector<String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void  YoloV3Detect::postprocess(const Mat& frame, const vector<Mat>& outs,
	vector<Rect>& predictROIs, vector<float>& predictScores, vector<string> &predictLabels)
{
	predictROIs.clear();
	predictScores.clear();
	predictLabels.clear();

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > m_confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices); // 非极大值抑制
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		string predictLabel = m_classes[classIds[idx]];

		predictLabels.push_back(predictLabel);
		predictROIs.push_back(box);
		predictScores.push_back(confidences[idx]);		
	}
}

YoloV3Detect::~YoloV3Detect()
{ 
	m_classes.clear(); 
}

