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
    
    auto type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t>input_node_dims = tensor_info.GetShape();
    
    int input_tensor_size = modelBlockSize;
    input_tensor_values = std::vector<float>(modelBlockSize);
    
    const size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
    }
    
    const size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
    }
    
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                   input_tensor_values.data(),
                                                   input_tensor_size,
                                                   input_node_dims.data(),
                                                   4);
    
    modelBlockSize = modelBlockSize; // maybe we can assign block size from loaded model specs
    modelSampleRate = modelSampleRate;
}

void OnnxModel::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // if DAW sample rate changes, we need to do down sample to 44100, or modelSampleRate
    currentSampleRate= sampleRate;
    currentBlockSize = samplesPerBlock;
}

void OnnxModel::reset()
{
    lagrangeInterpolator.reset();
    fifoData = std::queue<float>(); //clear fifo
}

void OnnxModel::downsample(AudioBuffer<float> &buffer)
{
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    if (currentSampleRate != modelSampleRate)
    {
        float resampleRatio = currentSampleRate/(float)modelSampleRate;
        
        if (isStereo)
        {
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(1),
                                         buffer.getWritePointer(1),
                                         buffer.getNumSamples());
        }
        else
        {
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
        }
    }
}

void OnnxModel::upsample(AudioBuffer<float> &buffer)
{
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    if (currentSampleRate != modelSampleRate)
    {
        float resampleRatio = modelSampleRate/(float)currentSampleRate;
        
        if (isStereo)
        {
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(1),
                                         buffer.getWritePointer(1),
                                         buffer.getNumSamples());
        }
        else
        {
            lagrangeInterpolator.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
        }
    }
}


void OnnxModel::process(AudioBuffer<float>& buffer)
{
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    downsample(buffer);

    // convert to mono if stereo
    if (isStereo)
    {
        // add the right (1) to the left (0)
        buffer.addFrom(0, 0, buffer, 1, 0, buffer.getNumSamples());
//        // copy the combined left (0) to the right (1)
//        buffer.copyFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());
    }
    
    // Store into FIFO
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
            input_tensor_values[i] = fifoData.front();
            fifoData.pop();
        }
        
        auto output_tensors =
              session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        
        
        for (auto i = 0; i < modelBlockSize; ++i)
        {
            fetchedData.push_back(floatarr[i]);
        }
        // transfer data to processed vector
        processedData.insert(std::end(processedData),
                             std::begin(fetchedData),
                             std::end(fetchedData));
    }
    
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
    
    // up sample if needed
    upsample(buffer);
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

