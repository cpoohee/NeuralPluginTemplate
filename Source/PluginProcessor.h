#pragma once

#include <JuceHeader.h>
#include "OnnxModel.h"
//==============================================================================
/**
*/
class NeuralDoublerAudioProcessor  : public juce::AudioProcessor
                            #if JucePlugin_Enable_ARA
                             , public juce::AudioProcessorARAExtension
                            #endif
{
public:
    //==============================================================================
    NeuralDoublerAudioProcessor();
    ~NeuralDoublerAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    
    void reset() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif
    
    void processBlock (AudioBuffer<float>& buffer, MidiBuffer& midiMessages) override;
    void processBlock (AudioBuffer<double>& buffer, MidiBuffer& midiMessages) override;
    
    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;
    
    // Our plug-in's current state
    juce::AudioProcessorValueTreeState state;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    
    std::vector<float> getInputRMSValue();
    std::vector<float> getOutputRMSValue();

private:
    //==============================================================================
    OnnxModel onnxModel;
    
    template <typename FloatType>
    void applyModel(AudioBuffer<FloatType>& buffer);
    
    template <typename FloatType>
    void process (AudioBuffer<FloatType>& buffer, AudioBuffer<FloatType>& dry_buffer);
    
    template <typename FloatType>
    void applyGain (AudioBuffer<FloatType>& buffer, float gainLevel);
    
    template <typename FloatType>
    void applyMixing(AudioBuffer<FloatType>& buffer, AudioBuffer<FloatType>& dryBuffer, float mix);
    
    std::vector<juce::LinearSmoothedValue<float>> inputRMS, outputRMS;
    void resetMeterValues();
    
    static BusesProperties getBusesProperties(){
        return BusesProperties().withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                                .withOutput ("Output", juce::AudioChannelSet::stereo(), true);
    }
    
    AudioBuffer<float> dryBuffer_float;
    AudioBuffer<double> dryBuffer_double;

    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralDoublerAudioProcessor)
};
