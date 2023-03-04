# pragma once

#include <JuceHeader.h>
#include "onnxruntime_cxx_api.h"
#include <type_traits>

class OnnxModel
{
public:
    OnnxModel();
    ~OnnxModel();
    
    void setup(File modelPath, int modelSampleRate, int modelBlockSize);
    
    void prepareToPlay (double sampleRate, int samplesPerBlock);
    
    void process(AudioBuffer<float>& buffer);
    void process(AudioBuffer<double>& buffer);
    
private:
    Ort::Env env;
    Ort::Session session{nullptr};
    
    int modelBlockSize;
    int modelSampleRate;
    
    int currentBlockSize;
    int currentSampleRate;
    
    std::queue<float> fifoData;
    
    
//    Ort::Value input_tensor_{nullptr};
//    std::array<float32_t, 3> input_shape_{1, 1, blocksize};
//
//    Ort::Value output_tensor_{nullptr};
//    std::array<float32_t, 3> output_shape_{1, 1, blocksize};
};
