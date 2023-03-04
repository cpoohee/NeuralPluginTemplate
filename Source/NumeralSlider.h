#pragma once

#include <JuceHeader.h>

class NumeralSlider : public juce::Slider
{
public:
    NumeralSlider() {}
    
    NumeralSlider(int dec) {
        decimalplaces = std::max(0, dec);
    }
    
    double getValueFromText (const juce::String& text) override
    {
        auto numeralText = text.trim();
        return numeralText.getDoubleValue();
    }

    juce::String getTextFromValue (double value) override
    {
        return juce::String(value, decimalplaces);
    }
    
private:
    int decimalplaces = 1;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NumeralSlider)
};
