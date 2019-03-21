#include "InferencePerformer.h"

using namespace cv;
using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
namespace fs = boost::filesystem;


void crop(Mat &img, double alpha, double beta)
{
    auto sz = img.size();
    int w = sz.width * alpha;
    int x = (sz.width - w) / 2;
    int h = sz.height * beta;
    int y = (sz.height - h) / 2;
    img = img(Rect(x, y, w, h));
}


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
        // if (severity != Severity::kINFO)
        
        cout << msg << std::endl;
    }
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

Mat InferencePerformer::vec2Mat(Mat& img)
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

InferencePerformer::InferencePerformer(const string &modelName, double ratio, double alpha) : ratio(ratio), alpha(alpha)
{
    i_ms = 0;
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

void InferencePerformer::postprocess(Mat &mask, bool deleteComponents, int &numOfComponents)
{
    mask.convertTo(mask, CV_8U);
    mask = 255 * mask;

    Mat components = Mat(mask.size(), CV_32S);
    int area = mask.size().height * mask.size().width;
    Mat stats, centroids;

    int nLabels = connectedComponentsWithStats(mask, components, stats, centroids, 4, CV_32S);
    numOfComponents = nLabels - 1;

    if (deleteComponents)
    {
        for (int label = 1; label < nLabels; label++)
        {
            if ((stats.at<int>(label, CC_STAT_AREA)) > ratio * area)
                continue;

            mask.setTo(Scalar(0.0), components == label);
            numOfComponents--;
        }
    }
}

void InferencePerformer::blend(Mat &img, Mat &mask)
{
    Mat tmp = img.clone();
    tmp.setTo(Scalar(255, 0, 0), mask);
    img = (1 - alpha) * img + alpha * tmp;
}

Mat InferencePerformer::getMask(Mat img, bool deleteComponents, int &numOfComponents)
{
    preprocess(img);
    Mat batch = vec2Mat(img);
    doInference((float*)batch.data);
    restoreOutput();
    Mat mask = (ch[1] > ch[0]);
    postprocess(mask, deleteComponents, numOfComponents);
    
    return mask;
}

Mat InferencePerformer::getMask(Mat &img, bool deleteComponents)
{
    int numOfComponents;
    return getMask(img, deleteComponents, numOfComponents);
}

void InferencePerformer::performInference(Mat &img, int &numOfComponents)
{
    Mat mask = getMask(img, true, numOfComponents);
    resize(img, img, displaySize, INTER_LINEAR);
    resize(mask, mask, displaySize, INTER_NEAREST);
    blend(img, mask);
}

void InferencePerformer::performInference(Mat &img)
{
    int numOfComponents;
    performInference(img, numOfComponents);
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
