#pragma once

#include <JuceHeader.h>

class DecibelSlider : public juce::Slider
{
public:
    DecibelSlider() {}
    
    DecibelSlider(int dec) {
        decimalplaces = std::max(0, dec);
        setSkewFactor(1000);
    }

    double getValueFromText (const juce::String& text) override{

        auto decibelText = text.upToFirstOccurrenceOf ("dB", false, false).trim();

        return decibelText.equalsIgnoreCase ("-INF") ? minusInfinitydB
                                                     : decibelText.getDoubleValue();
    }

    juce::String getTextFromValue (double value) override{
        return  juce::Decibels::toString (value, decimalplaces);
    }

private:
    int decimalplaces = 1;
    const float minusInfinitydB = -100.0;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DecibelSlider)
};
