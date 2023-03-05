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
    
    void reset();
    
private:
    Ort::Env env;
    Ort::Session session{nullptr};
    
    int modelBlockSize_;
    int modelSampleRate_;
    
    int currentBlockSize_;
    int currentSampleRate_;
    
    std::queue<float> fifoData;
    Interpolators::Lagrange lagrangeInterpolator;
    
    void downsample(AudioBuffer<float> &buffer);
    void upsample(AudioBuffer<float> &buffer);
    
    Ort::Value input_tensor{nullptr};
    Ort::Value output_tensor{nullptr};
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<float> input_tensor_values;
    std::vector<int64_t>input_node_dims;
};
