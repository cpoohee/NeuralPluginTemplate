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


void OnnxModel::process(AudioBuffer<float>& buffer)
{
    // TODO: down sample if needed
    
    // Store into FIFO
    
    // convert to mono if stereo
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    if (isStereo)
    {
        // add the right (1) to the left (0)
        buffer.addFrom(0, 0, buffer, 1, 0, buffer.getNumSamples());
//        // copy the combined left (0) to the right (1)
//        buffer.copyFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());
    }
    
    const float * readBuf = buffer.getReadPointer(0);
    int bufSize = buffer.getNumSamples();
    for (auto i = 0; i < bufSize; ++i)
    {
        fifoData.push(readBuf[i]);
    }
    
    std::vector<float> processedData;
    
    // when fifo buffer is large enough to do processing
    while (fifoData.size() >= modelBlockSize)
    {
        std::vector<float> fetchedData;
        for (auto i = 0; i < modelBlockSize; ++i)
        {
            fetchedData.push_back(fifoData.front());
            fifoData.pop();
        }
        
        // fake doing stuff
        processedData.insert(std::end(processedData),
                             std::begin(fetchedData),
                             std::end(fetchedData));
    }
    
    // TODO: up sample if needed
    
    
    buffer.clear();
    // write back to audio buffer
    if (processedData.size() > 0)
    {
        int channelSize = (isStereo)? 2 : 1;
            
        buffer.setSize(channelSize, static_cast<int>(processedData.size()));
        buffer.copyFrom(0, 0, &processedData[0], static_cast<int>(processedData.size()));
        
        if (isStereo){
            // copy left to right
            buffer.copyFrom(0, 0, buffer, 1, 0, buffer.getNumSamples());
        }
    }
}

void OnnxModel::process(AudioBuffer<double>& buffer)
{
    // convert to float
    AudioBuffer<float> temp_buffer;
    temp_buffer.makeCopyOf(buffer);
    process(temp_buffer);

    // clear old buffer, not ideal for now..
    int orgNumSamples = buffer.getNumSamples();
    for (auto i = 0; i< buffer.getNumChannels(); ++i)
        buffer.clear (i, 0, orgNumSamples);
    
    // convert to back to double
    buffer.makeCopyOf(temp_buffer);
}

