#include "PluginProcessor.h"
#include "PluginEditor.h"


//==============================================================================
NeuralDoublerAudioProcessor::NeuralDoublerAudioProcessor()
: AudioProcessor (getBusesProperties()),
                state (*this, nullptr, "state",
                       { std::make_unique<AudioParameterFloat> (ParameterID { "preGain",  1 }, "Input",  NormalisableRange<float> (-100.0f, 12.0f, 0.1f, 4.f), 0.0f),  // start, end, interval, skew
                         std::make_unique<AudioParameterFloat> (ParameterID { "postGain", 1 }, "Output", NormalisableRange<float>(-100.0f, 12.0f, 0.1f, 4.f), 0.0f),
                         std::make_unique<AudioParameterFloat> (ParameterID { "mix", 1 }, "Wet/Dry", NormalisableRange<float> (0.0f, 100.0f, 0.1f), 100.0f),
                    
                })
{
    auto bundle = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory().getParentDirectory();
    // to update these params for different models
    auto model_file = bundle.getChildFile ("Resources/model/aw_wavenet.onnx");
    int modelSampleRate = 44100;
    int modelBlockSize = 1024;
    
    onnxModel.setup(model_file, modelSampleRate, modelBlockSize);
    
    // Add a sub-tree to store the state of our UI
    state.state.addChild ({ "uiState", { { "width",  600 }, { "height", 450 } }, {} }, -1, nullptr);

    resetMeterValues();
}

NeuralDoublerAudioProcessor::~NeuralDoublerAudioProcessor()
{
}

//==============================================================================
const juce::String NeuralDoublerAudioProcessor::getName() const
{
    return "Neural Doubler";
}

bool NeuralDoublerAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool NeuralDoublerAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool NeuralDoublerAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double NeuralDoublerAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int NeuralDoublerAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int NeuralDoublerAudioProcessor::getCurrentProgram()
{
    return 0;
}

void NeuralDoublerAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String NeuralDoublerAudioProcessor::getProgramName (int index)
{
    return {};
}

void NeuralDoublerAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void NeuralDoublerAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // reinitilise meter
    for (auto i = 0; i < inputRMS.size(); ++i)
        inputRMS[i].reset(sampleRate, 0.5f);
                    
    for (auto i = 0; i < outputRMS.size(); ++i)
        outputRMS[i].reset(sampleRate, 0.5f);
    
    if (isUsingDoublePrecision())
    {
        dryBuffer_double.setSize (getTotalNumInputChannels(), samplesPerBlock);
        dryBuffer_float .setSize (1, 1);
    }
    else
    {
        dryBuffer_float.setSize (getTotalNumInputChannels(), samplesPerBlock);
        dryBuffer_double.setSize (1, 1);
    }
    
    onnxModel.prepareToPlay(sampleRate, samplesPerBlock);
}

void NeuralDoublerAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

void NeuralDoublerAudioProcessor::reset()
{
    // reset temp buffers
    dryBuffer_float.clear();
    dryBuffer_double.clear();
    
    // reset meter values
    resetMeterValues();
    
    // resets onnxmodel fifo and interpolator
    onnxModel.reset();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool NeuralDoublerAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void NeuralDoublerAudioProcessor::processBlock (AudioBuffer<float>& buffer, MidiBuffer& midiMessages)
{
    jassert (! isUsingDoublePrecision());
    process (buffer, dryBuffer_float, dryQueue_float);
}

void NeuralDoublerAudioProcessor::processBlock (AudioBuffer<double>& buffer, MidiBuffer& midiMessages)
{
    jassert (isUsingDoublePrecision());
    process (buffer, dryBuffer_double, dryQueue_double);
}

template <typename FloatType>
void NeuralDoublerAudioProcessor::process(juce::AudioBuffer<FloatType>& buffer,
                                          AudioBuffer<FloatType>& dry_buffer,
                                          std::queue<FloatType>& dryQueue)
{
    // deal with it
    if (buffer.getNumSamples() == 0){
        return; // there is nothing to do
    }
    
    //Returns 0 to 1 values
    auto preGainRawValue  = state.getParameter ("preGain")->getValue();
    auto postGainRawValue = state.getParameter ("postGain")->getValue();
    auto mixRawValue = state.getParameter("mix")->getValue();
    
    // convert back to representation
    auto preGainParamValue  = state.getParameter ("preGain")->convertFrom0to1(preGainRawValue);
    auto postGainParamValue  = state.getParameter ("postGain")->convertFrom0to1(postGainRawValue);
    auto mixParamValue  = mixRawValue;
    
    auto numSamples = buffer.getNumSamples();

    // In case we have more outputs than inputs, we'll clear any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear (i, 0, numSamples);
    
    // Apply our gain change to the outgoing data..
    applyGain (buffer, Decibels::decibelsToGain(preGainParamValue));
    
    // calculate input rms for metering
    for (auto i = 0; i < getTotalNumInputChannels(); ++i)
    {
        const auto channelRMS =  static_cast<float>(Decibels::gainToDecibels(buffer.getRMSLevel(i, 0, numSamples)));
        
        inputRMS[i].skip(numSamples);
        
        if (channelRMS < inputRMS[i].getCurrentValue())
        {
            inputRMS[i].setTargetValue(channelRMS); // smooth down
        }
        else
        {
            inputRMS[i].setCurrentAndTargetValue(channelRMS); // immediate set to target
        }
    }
        
    // Dry data sent to fifo buffer
    bool isStereo = (buffer.getNumChannels()>=2)? true:false;
    
    if (isStereo)
    {
        // only push left or mono channel
        const FloatType * readBuf1 = buffer.getReadPointer(0);
        const FloatType * readBuf2 = buffer.getReadPointer(1);
        int bufSize = buffer.getNumSamples();
        for (auto i = 0; i < bufSize; ++i)
        {
            dryQueue.push(readBuf1[i] + readBuf2[i]); // sum up
        }
    }
    else{
        const FloatType * readBuf = buffer.getReadPointer(0);
        int bufSize = buffer.getNumSamples();
        for (auto i = 0; i < bufSize; ++i)
        {
            dryQueue.push(readBuf[i]);
        }
    }

    // process wet
    applyModel(buffer);
    
    // update new number of samples
    numSamples = buffer.getNumSamples();
    
    // pop number of samples from fifo dry buffer, match with sample size after model output
    int channelSize = (isStereo)? 2 : 1;
    dry_buffer.setSize(channelSize, numSamples);
    if (isStereo)
    {
        for (int i = 0; i < numSamples; ++i){
            dry_buffer.setSample(0, i, dryQueue.front());
            dry_buffer.setSample(1, i, dryQueue.front());
            dryQueue.pop();
        }
    }
    else
    {
        for (int i = 0; i < numSamples; ++i){
            dry_buffer.setSample(0, i, dryQueue.front());
            dryQueue.pop();
        }
    }
    
    
    // mix wet and dry
    applyMixing(buffer, dry_buffer, mixParamValue);
        
    // apply output gain
    applyGain (buffer, Decibels::decibelsToGain(postGainParamValue));
    
    // calculate output rms for metering
    for (auto i = 0; i < buffer.getNumChannels(); ++i)
    {
        const auto channelRMS =  static_cast<float>(Decibels::gainToDecibels(buffer.getRMSLevel(i, 0, numSamples)));
        
        outputRMS[i].skip(numSamples);
        
        if (channelRMS < outputRMS[i].getCurrentValue())
        {
            outputRMS[i].setTargetValue(channelRMS); // smooth down
        }
        else
        {
            outputRMS[i].setCurrentAndTargetValue(channelRMS); // immediate targt
        }
    }
}

//==============================================================================
bool NeuralDoublerAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* NeuralDoublerAudioProcessor::createEditor()
{
    return new NeuralDoublerAudioProcessorEditor (*this);
}

//==============================================================================
void NeuralDoublerAudioProcessor::getStateInformation (MemoryBlock& destData)
{
    // Store an xml representation of our state.
    if (auto xmlState = state.copyState().createXml())
        copyXmlToBinary (*xmlState, destData);
}

void NeuralDoublerAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // Restore our plug-in's state from the xml representation stored in the above
    // method.
    if (auto xmlState = getXmlFromBinary (data, sizeInBytes))
        state.replaceState (ValueTree::fromXml (*xmlState));
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralDoublerAudioProcessor();
}

void NeuralDoublerAudioProcessor::resetMeterValues()
{
    inputRMS.clear();
    outputRMS.clear();
    
    for (auto i = 0; i < getTotalNumInputChannels(); ++i)
    {
        juce::LinearSmoothedValue<float> val;
        val.setTargetValue(-100.0f);
        inputRMS.push_back(val);
    }
                    
    for (auto i = 0; i < getTotalNumOutputChannels(); ++i)
    {
        juce::LinearSmoothedValue<float> val;
        val.setTargetValue(-100.0f);
        outputRMS.push_back(val);
    }
}

template <typename FloatType>
void NeuralDoublerAudioProcessor::applyModel(AudioBuffer<FloatType>& buffer)
{
    onnxModel.process(buffer);
}

template <typename FloatType>
void NeuralDoublerAudioProcessor::applyMixing(AudioBuffer<FloatType>& buffer, AudioBuffer<FloatType>& dryBuffer, float mix)
{
    // use linear
    auto dryValue = (1.0f) - mix;
    auto wetValue = mix;

    buffer.applyGain(wetValue);
    dryBuffer.applyGain(dryValue);
    
    for (auto i = 0; i < getTotalNumOutputChannels(); ++i)
        buffer.addFrom(i, 0, dryBuffer, i, 0, buffer.getNumSamples());
}

template <typename FloatType>
void NeuralDoublerAudioProcessor::applyGain (AudioBuffer<FloatType>& buffer, float gainLevel)
{
    for (auto channel = 0; channel < getTotalNumOutputChannels(); ++channel)
        buffer.applyGain (channel, 0, buffer.getNumSamples(), gainLevel);
}

std::vector<float> NeuralDoublerAudioProcessor::getInputRMSValue()
{
    std::vector<float> rmsValues;
    for (auto i : inputRMS)
    {
        rmsValues.push_back(i.getCurrentValue());
    }
    return rmsValues;
}

std::vector<float> NeuralDoublerAudioProcessor::getOutputRMSValue()
{
    std::vector<float> rmsValues;
    for (auto i : outputRMS)
    {
        rmsValues.push_back(i.getCurrentValue());
    }
    return rmsValues;
}
