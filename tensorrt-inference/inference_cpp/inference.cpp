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


int inputIndex, outputIndex;
nvinfer1::DimsCHW inputDims, outputDims;


#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        //if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


cv::Mat vec2Mat(cv::Mat& img)
{
    Mat batch;
    if (!img.empty()) { 
        const int width = img.cols;
        const int height = img.rows;
        const ptrdiff_t batchSize(1);
        const int batch_channels = img.channels();

        batch = Mat(height, width, CV_MAKETYPE(CV_32F, batch_channels));
        float* input_data = reinterpret_cast<float*>(batch.data);
        vector<Mat> ch;
        ch.reserve(img.channels());
        for (ptrdiff_t iSample = 0; iSample < batchSize; iSample += 1) {
            ch.resize(0);
            for (ptrdiff_t iCh = 0; iCh < img.channels(); iCh += 1) {
                Mat ttt = Mat(height, width, CV_32FC1, input_data);
                ch.push_back(Mat(height, width, CV_32FC1, input_data));
                input_data += height * width;
            }
            split(img, ch);
        }
    }
    return batch;
}


ICudaEngine * readTensorRTModel(const string& name)
{
    ICudaEngine *engine;
    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(name, std::ios::binary);
    if (file.good()) {
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
    if (gieModelStream) delete [] gieModelStream;

    for (int b = 0; b < engine -> getNbBindings(); ++b)
    {
        if (engine -> bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    inputDims = static_cast<nvinfer1::DimsCHW  &&>(engine -> getBindingDimensions(inputIndex));
    outputDims = static_cast<nvinfer1::DimsCHW  &&>(engine -> getBindingDimensions(outputIndex));

    return engine;
}


void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);

    void* buffers[2];
    int inputIndex, outputIndex;

    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    size_t inputSize = batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
    size_t outputSize = batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}



//             ******************** HARDCODE INSIDE ********************

int main() {

    string modelName = fs::canonical("../unet.engine").string();
    //string imagePath = fs::canonical("../input/test.jpg").string();
    string outImgPath = fs::canonical("../output/").string();
    
    //int N = 3;

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    //===============================================================================


    ICudaEngine *engine(readTensorRTModel(modelName));
    IExecutionContext *context = engine->createExecutionContext();

    int BUFF_SIZE = outputDims.c() * outputDims.h() * outputDims.w();
    float *buf = new float[BUFF_SIZE];
    
    long long ms = 0;
    

    //##########################################################
    //###########       PREPROCESSING              #############


    imshow("", 0);
    Mat img, resized, batch, outImage; //(imread(imagePath, CV_32F));
    vector<Mat> ch;
    namedWindow("segm", 1);

    for (;;)
    {

        //##########################################################
        //###########           INFERENCE              #############

        //auto t_start = std::chrono::high_resolution_clock::now();
        cap >> img;
	img.convertTo(img, CV_32FC3);
        cv::resize(img, resized, Size(inputDims.w(), inputDims.h()), INTER_AREA);
	resized /= 255.0;
	batch = vec2Mat(resized);

	doInference(*context, (float*)batch.data, buf, 1);


        
	//##########################################################
        //###########          POSTPROCESSING           ############

	float *resutlData = buf;

    	for(int i = 0; i < outputDims.c(); i++)
    	{
        	cv::Mat iclass(Size(outputDims.w(), outputDims.h()), CV_32FC1, resutlData);
        	ch.push_back(iclass);
        	resutlData += outputDims.h() * outputDims.w();
    	}

        outImage = Mat(resized);
	outImage.setTo(cv::Scalar(0, 255, 0), ch[1] > ch[0]);
        cv::resize(outImage, img, Size(2 * inputDims.w(), 2 * inputDims.h()), INTER_AREA);

        //auto t_end = std::chrono::high_resolution_clock::now();
        //ms += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

	ch.clear();

	
	imshow("segm", img);
	
	if(waitKey(30) >= 0) break;
	
    	
	//imwrite(outImgPath + "/img.png", outImage);
    }

    context->destroy();
    engine->destroy();

    //cout << "avg inference time " << 1e-6 * ms / N << " ms" << endl;

    return 0;
}
