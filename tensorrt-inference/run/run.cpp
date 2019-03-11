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

using namespace cv;
using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
namespace fs = boost::filesystem;


#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}


long long i_ms = 0;


cv::Mat vec2Mat(cv::Mat& img)
{
    Mat batch;

    if (!img.empty())
    { 
        const int width = img.cols;
        const int height = img.rows;
        const ptrdiff_t batchSize(1);
        const int batch_channels = img.channels();

        batch = Mat(height, width, CV_MAKETYPE(CV_32F, batch_channels));
        float* input_data = reinterpret_cast<float*>(batch.data);
        vector<Mat> ch;
        ch.reserve(img.channels());
        for (ptrdiff_t iSample = 0; iSample < batchSize; iSample += 1)
        {
            ch.resize(0);
            for (ptrdiff_t iCh = 0; iCh < img.channels(); iCh += 1)
            {
                ch.push_back(Mat(height, width, CV_32FC1, input_data));
                input_data += height * width;
            }
            split(img, ch);
        }
    }

    return batch;
}


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        // if (severity != Severity::kINFO)
        
        std::cout << msg << std::endl;
    }
};


class InferencePerformer
{
    public:
        Size inferenceSize, displaySize;
	Mat getMask(Mat img, bool deleteComponents);
	void performInference(Mat &img);
	InferencePerformer(const string &modelName, double ratio=0.0003, double alpha=0.33);
	~InferencePerformer();
    private:
	double ratio, alpha;
	vector<Mat> ch;
	float *outputBuffer;
	void *cudaBuffers[2];
	cudaStream_t stream;
	ICudaEngine *engine;
	IExecutionContext *context;
	int inputIndex, outputIndex;
	DimsCHW inputDims, outputDims;
	size_t inputSize, outputSize;
	void readTensorRTModel(const string &modelName);
	void initBuffers();
	void doInference(float *input);
	void restoreOutput();
	void preprocess(Mat &img);
	void delSmallComponents(Mat &mask);
	void postprocess(Mat &mask, bool deleteComponents);
	void blend(Mat &img, Mat &mask);
};


void InferencePerformer::readTensorRTModel(const string& modelName)
{
    Logger gLogger;
    char *gieModelStream{nullptr};
    size_t size{0};
    ifstream file(modelName, ios::binary);
    
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        gieModelStream = new char[size];
        assert(gieModelStream);
        file.read(gieModelStream, size);
        file.close();
    }
    
    IRuntime* infer = createInferRuntime(gLogger);
    engine = infer->deserializeCudaEngine(gieModelStream, size, nullptr);
    if (gieModelStream)
        delete [] gieModelStream;
}

void InferencePerformer::initBuffers()
{
    assert(engine->getNbBindings() == 2);

    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        if (engine->bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    inputDims = static_cast<DimsCHW &&>(engine->getBindingDimensions(inputIndex));
    outputDims = static_cast<DimsCHW &&>(engine->getBindingDimensions(outputIndex));
    inferenceSize = Size(inputDims.w(), inputDims.h());
    displaySize = Size(1152, 624);
    
    int buf_size = outputDims.c() * outputDims.w() * outputDims.h();
    outputBuffer = (float*)calloc(buf_size, sizeof(float));
    
    inputSize = 1 * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
    outputSize = 1 * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

    CHECK(cudaMalloc(&cudaBuffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&cudaBuffers[outputIndex], outputSize));

    CHECK(cudaStreamCreate(&stream));
}

InferencePerformer::InferencePerformer(const string &modelName, double ratio, double alpha) : ratio(ratio), alpha(alpha)
{
    readTensorRTModel(modelName);
    context = engine->createExecutionContext();
    initBuffers();

    cout << endl;
    cout << "model name: " << modelName << endl;
    cout << "inference resolution: " << inferenceSize << endl;
}

void InferencePerformer::doInference(float* input)
{
    auto t_start = chrono::high_resolution_clock::now();

    CHECK(cudaMemcpyAsync(cudaBuffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueue(1, cudaBuffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputBuffer, cudaBuffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    auto t_end = chrono::high_resolution_clock::now();
    i_ms += chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
}

void InferencePerformer::restoreOutput()
{
    ch.clear();
    float *resultData = outputBuffer;

    for (int i = 0; i < outputDims.c(); i++)
    {
	Mat iclass(Size(outputDims.w(), outputDims.h()), CV_32FC1, resultData);
	ch.push_back(iclass);
	resultData += outputDims.h() * outputDims.w();
    }
}

void InferencePerformer::preprocess(Mat &img)
{
    img.convertTo(img, CV_32FC3);
    resize(img, img, inferenceSize, INTER_LINEAR);
    img /= 255.0;
}

void InferencePerformer::delSmallComponents(Mat &mask)
{
    Mat components = Mat(mask.size(), CV_32S);
    int area = mask.size().height * mask.size().width;
    Mat stats, centroids;

    int nLabels = connectedComponentsWithStats(mask, components, stats, centroids, 4, CV_32S);

    for (int label = 1; label < nLabels; label++)
    {
        if ((stats.at<int>(label, CC_STAT_AREA)) > ratio * area)
            continue;

        mask.setTo(Scalar(0.0), components == label);
    }
}

void InferencePerformer::postprocess(Mat &mask, bool deleteComponents)
{
    mask.convertTo(mask, CV_8U);
    mask = 255 * mask;

    if (deleteComponents)
        delSmallComponents(mask);
}

void InferencePerformer::blend(Mat &img, Mat &mask)
{
    Mat tmp = img.clone();
    tmp.setTo(Scalar(255, 0, 0), mask);
    img = (1 - alpha) * img + alpha * tmp;
}


Mat InferencePerformer::getMask(Mat img, bool deleteComponents)
{
    preprocess(img);
    Mat batch = vec2Mat(img);
    
    doInference((float*)batch.data);
    restoreOutput();
    Mat mask = (ch[1] > ch[0]);
    postprocess(mask, deleteComponents);
    
    return mask;
}

void InferencePerformer::performInference(Mat &img)
{
    Mat mask = getMask(img, true);
    resize(img, img, displaySize, INTER_LINEAR);
    resize(mask, mask, displaySize, INTER_NEAREST);
    blend(img, mask);
}


InferencePerformer::~InferencePerformer()
{
    free(outputBuffer);
   
    cudaStreamDestroy(stream);
    CHECK(cudaFree(cudaBuffers[inputIndex]));
    CHECK(cudaFree(cudaBuffers[outputIndex]));

    context->destroy();
    engine->destroy();
}


void crop(Mat &img, double alpha, double beta)
{
    auto sz = img.size();
    int w = sz.width * alpha;
    int x = (sz.width - w) / 2;
    int h = sz.height * beta;
    int y = (sz.height - h) / 2;
    img = img(Rect(x, y, w, h));
}


int main(int argc, char *argv[])
{
    string modelName(argv[1]);
    InferencePerformer performer(modelName);


    char *junk_ptr;
    double alpha = 1.0;
    double beta = 1.0;

    if (argc == 4)
    {
	alpha = strtod(argv[2], &junk_ptr);
	beta = strtod(argv[3], &junk_ptr);
    }


    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    //cap.set(CAP_PROP_FPS, 100);

    Mat img;
    namedWindow("segm", 1);

    long long ms = 0, cap_ms = 0;
    int N = 0;
    //Mat img1 = imread("../../data/raw3/2018-11-20_133328_led.bmp", CV_LOAD_IMAGE_COLOR);
    //resize(img1, img1, Size(770, 420), INTER_LINEAR);


    auto t_start = chrono::high_resolution_clock::now();
    
    for (;; N++)
    {
    auto tt_start = chrono::high_resolution_clock::now();

        cap >> img;
        //img = img1.clone();
	crop(img, alpha, beta);

    auto tt_end = chrono::high_resolution_clock::now();
    cap_ms += chrono::duration_cast<std::chrono::nanoseconds>(tt_end - tt_start).count();

	performer.performInference(img);
	imshow("segm", img);
	
	if (waitKey(1) >= 0)
	    break;
    }

    auto t_end = chrono::high_resolution_clock::now();
    ms += chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

    cout << "avg cap time " << 1e-6 * cap_ms / N << " ms" << endl;
    cout << "avg inference time " << 1e-6 * i_ms / N << " ms" << endl;
    cout << "avg run time " << 1e-6 * ms / N << " ms" << endl;
    cout << endl;


    return 0;
}
