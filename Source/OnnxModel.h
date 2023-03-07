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
    Interpolators::Lagrange lagrangeInterpolatorUp1, lagrangeInterpolatorUp2;
    Interpolators::Lagrange lagrangeInterpolatorDown1, lagrangeInterpolatorDown2;
    
    void downsample(AudioBuffer<float> &buffer);
    void upsample(AudioBuffer<float> &buffer);
    
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t>input_node_dims;
    
    std::vector<float>prev_block;
};
