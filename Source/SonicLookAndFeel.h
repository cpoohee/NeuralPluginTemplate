#pragma once

#include <JuceHeader.h>

class SonicLookAndFeel : public juce::LookAndFeel_V4
{
public:
    SonicLookAndFeel()
    {
        // slider colour
        setColour(Slider::ColourIds::thumbColourId, Colours::black);
        setColour(Slider::ColourIds::rotarySliderFillColourId, Colours::darkslategrey);
        setColour(Slider::ColourIds::backgroundColourId, Colours::lightgrey);
        
        // text box
        setColour(Slider::ColourIds::textBoxTextColourId, Colours::black);
        setColour(Slider::ColourIds::textBoxOutlineColourId, Colours::lightgrey);
        
        // Label text colour
        setColour(Label::ColourIds::textColourId, Colours::black);
        
        // toggle button
        setColour(ToggleButton::ColourIds::textColourId, Colours::black);
        setColour(ToggleButton::ColourIds::tickColourId, Colours::black);
        setColour(ToggleButton::ColourIds::tickDisabledColourId, Colours::black);
    }

    // set slider thumb size
    int getSliderThumbRadius(Slider &slider) { return sliderThumbSize; };
    Font getTitleFont(){ return Font(titleFontTypeface,
                                     titleFontSize,
                                     Font::FontStyleFlags::bold);
    };
    int getTitleFontSize(){ return titleFontSize; };
    int getFontSize(){ return fontSize; };
        
private :
    const int fontSize = 15;
    const float titleFontSize = fontSize * 4.0f;
    const int sliderThumbSize = 28;
    const String titleFontTypeface = "Arial";

};
