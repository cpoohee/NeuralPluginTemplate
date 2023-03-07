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

    const size_t num_input_nodes = session.GetInputCount();
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));
        // print input node types
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
    }

    const size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        output_names_ptr.push_back(std::move(output_name));
    }
    
    modelBlockSize_ = modelBlockSize; // maybe we can assign block size from loaded model specs
    modelSampleRate_ = modelSampleRate;
}

void OnnxModel::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // if DAW sample rate changes, we need to do down sample to 44100, or modelSampleRate
    currentSampleRate_ = sampleRate;
    currentBlockSize_ = samplesPerBlock;
    prev_block = std::vector<float>(samplesPerBlock, 0.0f);
}

void OnnxModel::reset()
{
    fifoData = std::queue<float>(); //clear fifo
    
    lagrangeInterpolatorUp1.reset();
    lagrangeInterpolatorUp2.reset();
    lagrangeInterpolatorDown1.reset();
    lagrangeInterpolatorDown2.reset();
}

void OnnxModel::downsample(AudioBuffer<float> &buffer)
{
    if (buffer.getNumSamples() == 0){
        return;
    }
    
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    if (currentSampleRate_ != modelSampleRate_)
    {
        float resampleRatio = currentSampleRate_/(float)modelSampleRate_;
        
        int reducedSamplesNum = buffer.getNumSamples() * 1/resampleRatio;
        
        if (isStereo)
        {
            lagrangeInterpolatorDown1.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
            lagrangeInterpolatorDown2.process(resampleRatio,
                                         buffer.getReadPointer(1),
                                         buffer.getWritePointer(1),
                                         buffer.getNumSamples());
        }
        else
        {
            lagrangeInterpolatorDown1.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
        }
    }
}

void OnnxModel::upsample(AudioBuffer<float> &buffer)
{
    if (buffer.getNumSamples() == 0){
        return;
    }
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    if (currentSampleRate_ != modelSampleRate_)
    {
        float resampleRatio = modelSampleRate_/(float)currentSampleRate_;
        
        int increasedSamplesNum = buffer.getNumSamples() * 1/resampleRatio;
        
        if (isStereo)
        {
            lagrangeInterpolatorUp1.process(resampleRatio,
                                         buffer.getReadPointer(0),
                                         buffer.getWritePointer(0),
                                         buffer.getNumSamples());
            lagrangeInterpolatorUp2.process(resampleRatio,
                                         buffer.getReadPointer(1),
                                         buffer.getWritePointer(1),
                                         buffer.getNumSamples());
        }
        else
        {
            lagrangeInterpolatorUp1.process(resampleRatio,
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
//    downsample(buffer);

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
    int fifoSize = (int)fifoData.size(); // in case of next time, queue is accessed from other threads..
    
    while (fifoSize >= modelBlockSize_)
    {
        std::vector<float>input_tensor_values(modelBlockSize_*2);
        
        // primed the beginning of the input block with prev block or empty block
        std::copy(prev_block.begin(), prev_block.end(), input_tensor_values.begin());
        
        for (auto i = 0; i < modelBlockSize_; ++i)
        {
            input_tensor_values[modelBlockSize_ + i] = fifoData.front();
            fifoData.pop();
        }
        fifoSize = fifoSize - modelBlockSize_;
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // set the variable sized sample length
        input_node_dims.at(2) = (int64_t)modelBlockSize_*2;
        
        // send 2 blocks for inference
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                       input_tensor_values.data(),
                                                       modelBlockSize_*2,
                                                       input_node_dims.data(),
                                                       input_node_dims.size());
        
        auto output_tensors =
              session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        
        // assign prev_block as latest block
        prev_block = input_tensor_values;

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        
        // drop the first block, keep the 2nd block
        std::vector<float> fetchedData(modelBlockSize_);
        for (int i = 0; i < modelBlockSize_; ++i)
        {
            fetchedData[i] = (floatarr[modelBlockSize_ + i]);
        }

        // transfer data to processed vector
        processedData.insert(processedData.end(),
                             fetchedData.begin(),
                             fetchedData.end());
    }

    // write back to audio buffer
    if (processedData.size() > 0)
    {
        int channelSize = (isStereo)? 2 : 1;

        buffer.setSize(channelSize, static_cast<int>(processedData.size()));
        buffer.copyFrom(0, 0, &processedData[0], static_cast<int>(processedData.size()));

        if (isStereo){
            // copy left to right
            buffer.copyFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());
            buffer.applyGain(0.5);
        }
    }
    else{
        buffer.setSize(0, 0);
        if (isStereo)
        {
            buffer.setSize(1, 0);
        }
    }
    
    // up sample if needed
//    upsample(buffer);
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
