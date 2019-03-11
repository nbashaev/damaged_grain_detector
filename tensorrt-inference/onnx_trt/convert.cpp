#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <string.h>
#include <chrono>
#include <iterator>
#include <vector>
#include <algorithm>

#include "NvInfer.h"
#include "NvOnnxParser.h"


using namespace nvinfer1;


class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) override
    {
        std::cout << msg << std::endl;
    }
};


class Int8CacheCalibrator : public IInt8EntropyCalibrator
{
public:
    Int8CacheCalibrator(std::string cacheFile) : mCacheFile(cacheFile) {}
    virtual ~Int8CacheCalibrator() {}

    int getBatchSize() const override {return 1;}

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return false;
    }
  
    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;

        if (input.good())
        {
            std::copy(std::istream_iterator<char>(input),
            std::istream_iterator<char>(),
            std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) override {}

private:
    std::string mCacheFile;
    std::vector<char> mCalibrationCache;
};


void onnxToTRTModel(const std::string& modelFile, IHostMemory*& trtModelStream, Int8CacheCalibrator *calibrator)
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
    Logger gLogger;


    IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(modelFile.c_str(), verbosity))
    {
        std::string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }


    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    if (calibrator)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    parser->destroy();

    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}


void saveModel(IHostMemory* trtModelStream, const std::string& outFile)
{
    std::ofstream ofs(outFile, std::ios::binary);
    ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
    ofs.close();
}


int main(int argc, char *argv[])
{
    IHostMemory* trtModelStream{nullptr};
    Int8CacheCalibrator* calibrator{nullptr};

    std::string onnx_file(argv[1]);
    std::string modelName = onnx_file;
    std::string engine_file(onnx_file);
    engine_file.erase(engine_file.end() - 5, engine_file.end());

    if (argc == 3)
    {
        std::string cache_file(argv[2]);
        calibrator = new Int8CacheCalibrator(cache_file);
        engine_file += "_int8";
    }

    engine_file += ".engine";

    onnxToTRTModel(onnx_file, trtModelStream, calibrator);
    saveModel(trtModelStream, engine_file);

    return 0;
}
