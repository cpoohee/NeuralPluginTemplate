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
    
//    auto type_info = session.GetInputTypeInfo(0);
//    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//    input_node_dims = tensor_info.GetShape();
//
//    int input_tensor_size = modelBlockSize;
//    input_tensor_values = std::vector<float>(modelBlockSize);
//
//    const size_t num_input_nodes = session.GetInputCount();
//    for (size_t i = 0; i < num_input_nodes; ++i)
//    {
//        auto input_name = session.GetInputNameAllocated(i, allocator);
//        input_node_names.push_back(input_name.get());
//    }
//
//    const size_t num_output_nodes = session.GetOutputCount();
//    for (size_t i = 0; i < num_output_nodes; ++i)
//    {
//        auto output_name = session.GetOutputNameAllocated(i, allocator);
//        output_node_names.push_back(output_name.get());
//    }

//    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    input_tensor = Ort::Value::CreateTensor<float>(memory_info,
//                                                   input_tensor_values.data(),
//                                                   input_tensor_size,
//                                                   input_node_dims.data(),
//                                                   input_node_dims.size());
    
    modelBlockSize_ = modelBlockSize; // maybe we can assign block size from loaded model specs
    modelSampleRate_ = modelSampleRate;
}

void OnnxModel::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // if DAW sample rate changes, we need to do down sample to 44100, or modelSampleRate
    currentSampleRate_ = sampleRate;
    currentBlockSize_ = samplesPerBlock;
}

void OnnxModel::reset()
{
    lagrangeInterpolator.reset();
//    fifoData = std::queue<float>(); //clear fifo
}

void OnnxModel::downsample(AudioBuffer<float> &buffer)
{
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    //down sample if needed
    if (currentSampleRate_ != modelSampleRate_)
    {
        float resampleRatio = currentSampleRate_/(float)modelSampleRate_;
        
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
    if (currentSampleRate_ != modelSampleRate_)
    {
        float resampleRatio = modelSampleRate_/(float)currentSampleRate_;
        
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
        auto type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        const size_t num_input_nodes = session.GetInputCount();
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<const char*> input_node_names;
        input_names_ptr.reserve(num_input_nodes);
        input_node_names.reserve(num_input_nodes);
        std::vector<int64_t> input_node_dims;
        
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
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
        std::vector<const char*> output_node_names;
        for (size_t i = 0; i < num_output_nodes; ++i)
        {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());
            output_names_ptr.push_back(std::move(output_name));
        }
        
        std::vector<float> fetchedData;
        
        int input_tensor_size = modelBlockSize_;
        std::vector<float>input_tensor_values(modelBlockSize_);
        
        for (auto i = 0; i < modelBlockSize_; ++i)
        {
            input_tensor_values[i] = fifoData.front();
//            fetchedData.push_back(fifoData.front());// testing
            fifoData.pop();
        }
        fifoSize = fifoSize - modelBlockSize_;
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                       input_tensor_values.data(),
                                                       modelBlockSize_,
                                                       input_node_dims.data(),
                                                       input_node_dims.size());

        auto output_tensors =
              session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        for (int i = 0; i < modelBlockSize_; ++i)
        {
            fetchedData.push_back(floatarr[i]);
        }

        // transfer data to processed vector
        processedData.insert(processedData.end(),
                             fetchedData.begin(),
                             fetchedData.end());
    }

    //buffer.clear();
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

