#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "DecibelSlider.h"
#include "NumeralSlider.h"
#include "LevelMeter.h"
#include "SonicLookAndFeel.h"

//==============================================================================
/**
*/
class NeuralDoublerAudioProcessorEditor  : public AudioProcessorEditor,
                                           private Timer,
                                           private Value::Listener
{
public:
    NeuralDoublerAudioProcessorEditor (NeuralDoublerAudioProcessor&);
    ~NeuralDoublerAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    
    void timerCallback() override;
    void resetMeters();

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
//    NeuralDoublerAudioProcessor& audioProcessor;
    
    Label preGainLabel  { {}, "Input" },
          postGainLabel { {}, "Output" },
          mixLabel  { {}, "Wet/Dry" },
          titleLabel { {}, "Neural Doubler" }  ;

    DecibelSlider preGainSlider, postGainSlider;
    NumeralSlider mixSlider, curveSlider;
    AudioProcessorValueTreeState::SliderAttachment preGainAttachment, postGainAttachment,mixAttachment;

//    AudioProcessorValueTreeState::ButtonAttachment zeroClipButtonAttachment, bandSplitButtonAttachment;
    
    OwnedArray<Gui::LevelMeter> inputMeters;
    OwnedArray<Gui::LevelMeter> outputMeters;
    
    SonicLookAndFeel sonicLookAndFeel;
    Colour backgroundColour;

    // these are used to persist the UI's size - the values are stored along with the
    // filter's other parameters, and the UI component will update them when it gets
    // resized.
    Value lastUIWidth, lastUIHeight;
    
    // called when the stored window size changes
    void valueChanged (Value&) override
    {
        setSize (lastUIWidth.getValue(), lastUIHeight.getValue());
    }
    
    NeuralDoublerAudioProcessor& getProcessor() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralDoublerAudioProcessorEditor)
};
