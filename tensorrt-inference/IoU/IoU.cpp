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
    //resize(img, img, displaySize, INTER_LINEAR);
    //resize(mask, mask, displaySize, INTER_NEAREST);
    //blend(img, mask);
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



int main(int argc, char *argv[])
{
    string modelName1(argv[1]);
    string modelName2(argv[2]);
    InferencePerformer performer1(modelName1);
    InferencePerformer performer2(modelName2);

/*

    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;
*/
    //cap.set(CAP_PROP_FPS, 100);

    Mat img;

    double IoU = 0.0, I, U;
    double add;

    int N = 0;
    
    for (auto& p: fs::directory_iterator("../../data/raw2"))
    {
	if (p.path().stem().string().back() == 'v')
	    continue;

	img = imread(p.path().string(), CV_LOAD_IMAGE_COLOR);
	

	Mat mask1 = performer1.getMask(img, false) > 0;
	Mat mask2 = performer2.getMask(img, false) > 0;

	I = sum(mask1 & mask2)[0];
	U = sum(mask1 | mask2)[0];
	
	
        if (U > 0)
	    add = (U > 0.0) ? (I / U) : 1.0;
        cout << add << endl;
	IoU += add;

        N++;
    }

    IoU /= N;

    cout << endl << N << ' ' << IoU << endl;


    return 0;
}
