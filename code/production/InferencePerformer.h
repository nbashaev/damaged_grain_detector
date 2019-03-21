#include <iostream>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>


void crop(cv::Mat &img, double alpha, double beta);


class InferencePerformer
{
    public:
        long long i_ms;
        cv::Size inferenceSize, displaySize;
	cv::Mat getMask(cv::Mat img, bool deleteComponents, int &numOfComponents);
	cv::Mat getMask(cv::Mat &img, bool deleteComponents);
	void performInference(cv::Mat &img, int &numOfComponents);
	void performInference(cv::Mat &img);
	InferencePerformer(const std::string &modelName, double ratio=0.0003, double alpha=0.33);
	~InferencePerformer();
    private:
	double ratio, alpha;
	std::vector<cv::Mat> ch;
	float *outputBuffer;
	void *cudaBuffers[2];
	cudaStream_t stream;
	nvinfer1::ICudaEngine *engine;
	nvinfer1::IExecutionContext *context;
	int inputIndex, outputIndex;
	nvinfer1::DimsCHW inputDims, outputDims;
	size_t inputSize, outputSize;
	void readTensorRTModel(const std::string &modelName);
	void initBuffers();
	cv::Mat vec2Mat(cv::Mat& img);
	void doInference(float *input);
	void restoreOutput();
	void preprocess(cv::Mat &img);
	void postprocess(cv::Mat &mask, bool deleteComponents, int &numOfComponents);
	void blend(cv::Mat &img, cv::Mat &mask);
};
