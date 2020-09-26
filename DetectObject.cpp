
#include "YoloV3Detect.h"
#include "mex.h"
#include "matrix.h"
#include "opencv2/opencv.hpp"
 
YoloV3Detect DetectObj;

void checkInputs_init(int nrhs, const mxArray *prhs[])
{
 
	// Check number of inputs
	if (nrhs != 6)
	{
		mexErrMsgTxt("Incorrect number of inputs. Function expects 6 inputs.");
	}
}

void checkInputs_detect(int nrhs, const mxArray *prhs[])
{

	// Check number of inputs
	if (nrhs != 2)
	{
		mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
	}

	// Check image data type
	if (!mxIsChar(prhs[0]))
	{
	   mexErrMsgTxt("first param must be 'detect'");
	}

	if (!mxIsUint8(prhs[1]))
	{
		mexErrMsgTxt("sencond param must a image data, image must be UINT8.");
	}

	//const mwSize *dim = mxGetDimensions(prhs[1]);
	//mexPrintf("输入维度: %f,%f,%f\n", (float)dim[0], (float)dim[1], (float)dim[2]);
	return;
}
 
// matlab call HumanDetect('init',path1,path2,path3,threshold1,threshold2)
void init(const mxArray *prhs[])
{
	const char* str1 = mxArrayToString(prhs[1]);
	const char* str2 = mxArrayToString(prhs[2]);
	const char* str3 = mxArrayToString(prhs[3]);

	const String modelConfiguration = str1;
	const String modelBinary = str2;
	const string  classesFile = str3;
	float confThreshold = (float)mxGetScalar(prhs[4]); //0.05
	float nmsThreshold = (float)mxGetScalar(prhs[5]); //0.1
	 // 初始化
	DetectObj.init(modelConfiguration, modelBinary, classesFile, confThreshold, nmsThreshold);

	// Free memory allocated by mxArrayToString
	mxFree((void *)str1);
	mxFree((void *)str2);
	mxFree((void *)str3);
}
 
// matlab call this function must in this way:
// [predictROIs,predictScores,predictLabels] = HumanDetect('detect',imgdata)  
void detect(mxArray *plhs[], const mxArray *prhs[])
{
	uchar * inputPr = (uchar *)mxGetData(prhs[1]);
	const mwSize *dim = mxGetDimensions(prhs[1]);
	
	int m = dim[0];// rows
	int n = dim[1];// cols
	if (dim[2]!=3)
	{
		mexErrMsgTxt("image data must 3 channels!");
	}
	
	Mat image = cv::Mat::zeros(m, n, CV_8UC3);
	int number = 0;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			image.at<Vec3b>(i, j)[0] = inputPr[j*m + i + 2 * m*n]; // blue
			image.at<Vec3b>(i, j)[1] = inputPr[j*m + i+m*n];
			image.at<Vec3b>(i, j)[2] =  inputPr[j*m + i];
		}
	}
	if (image.empty())
	{
		mexErrMsgTxt("image is empty()！");
	}
	//vector<Rect> HumanROIs;
	//vector<float> scores;
	vector<Rect> predictROIs;
	vector<float> predictScores;
	vector<string> predictLabels;

	DetectObj.detect(image, predictROIs, predictScores, predictLabels);
	
	// mxArray convert to opencv Mat
	cv::Mat matOut = cv::Mat(predictROIs.size(), 4, CV_32FC1);
	for (size_t i = 0; i < (predictROIs).size(); i++)
	{
		float* data = matOut.ptr<float>(i);
		data[0] = (predictROIs)[i].x;
		data[1] = (predictROIs)[i].y;
		data[2] = (predictROIs)[i].width;
		data[3] = (predictROIs)[i].height;
	}

	// convert to mxArray 
	int rows = (predictROIs).size();
	int cols = 4;
	plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(rows, 1, mxREAL);
	
	double *temp,*temp2;
	temp = mxGetPr(plhs[0]);
	temp2 = mxGetPr(plhs[1]);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			*(temp + i + j * rows) = (double)matOut.at<float>(i, j);
		}
		*temp2 = (double)(predictScores)[i];
		temp2 ++ ;
	}		
	// "vector<string>" convert to  matlab "cell" type ,https://stackoverflow.com/questions/2867340/how-to-create-a-string-array-in-matlab/55369548#55369548
	mxArray *arr = mxCreateCellMatrix(rows, 1);
	for (mwIndex i = 0; i<rows; i++) {
		mxArray *str = mxCreateString(predictLabels[i].c_str());
		mxSetCell(arr, i, mxDuplicateArray(str));
		mxDestroyArray(str);
	}
	plhs[2] = arr;

}

void exitFcn()
{

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const char *str = mxIsChar(prhs[0]) ? mxArrayToString(prhs[0]) : NULL;
	// 
	if (str != NULL)
	{
		if (strcmp(str, "init") == 0)
		{
			checkInputs_init(nrhs, prhs);
			init(prhs);
		}
		else if (strcmp(str, "detect") == 0)
		{
			checkInputs_detect(nrhs, prhs);
			detect(plhs, prhs);
		}	
		else if (strcmp(str, "destroy") == 0)
			exitFcn();

		// Free memory allocated by mxArrayToString
		mxFree((void *)str);
	}
}
