#include "OnnxModel.h"

OnnxModel::OnnxModel()
{

}

OnnxModel::~OnnxModel()
{

}

void OnnxModel::setup(File modelPath, int modelSampleRate, int modelBlockSize)
{
    env = Ort::Env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default"));
    session = Ort::Session(env, modelPath.getFullPathName().toRawUTF8() , Ort::SessionOptions{nullptr});
    
    modelBlockSize = modelBlockSize; // to do.. assign block size from loaded model specs
    modelSampleRate = modelSampleRate;
}

void OnnxModel::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // if DAW sample rate changes, we need to do down sample to 44100, or modelSampleRate
    currentBlockSize = sampleRate;
    currentSampleRate = samplesPerBlock;
}

template <typename FloatType>
void process(const FloatType **inputData, FloatType **outputData, int numSamples)
{
    
}

void OnnxModel::process(const double **inputData, double **outputData, int numSamples)
{
    // todo.. convert to float
    //process()
    
    // todo.. convert to back to double
}

void OnnxModel::process(const float **inputData, float **outputData, int numSamples)
{
    // down sample if needed
    
    // Store into FIFO
    

    
    // up sample if needed
}
