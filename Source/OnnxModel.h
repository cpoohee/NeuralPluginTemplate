# pragma once

#include <JuceHeader.h>
#include "onnxruntime_cxx_api.h"

class OnnxModel
{
public:
    OnnxModel();
    ~OnnxModel();
    
    void setup(File modelPath, int modelSampleRate, int modelBlockSize);
    
    void prepareToPlay (double sampleRate, int samplesPerBlock);
    
    template <typename FloatType>
    void process(const FloatType **inputData, FloatType **outputData, int numSamples);
    void process(const double **inputData, double **outputData, int numSamples);
    void process(const float **inputData, float **outputData, int numSamples);
    
private:
    Ort::Env env;
    Ort::Session session{nullptr};
    
    int modelBlockSize;
    int modelSampleRate;
    
    int currentBlockSize;
    int currentSampleRate;
    
    AudioBuffer<float> fifoData;
    
//    Ort::Value input_tensor_{nullptr};
//    std::array<float32_t, 3> input_shape_{1, 1, blocksize};
//
//    Ort::Value output_tensor_{nullptr};
//    std::array<float32_t, 3> output_shape_{1, 1, blocksize};
};
