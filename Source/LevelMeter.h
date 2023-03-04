#pragma once

#include <JuceHeader.h>

namespace Gui {

    class LevelMeter : public Component{
    public:
        LevelMeter(){
        }
        
        LevelMeter(float min, float max){
            minRange = min;
            maxRange = max;
        }
        void paint(Graphics &g) override{
            auto bounds = getLocalBounds().toFloat();
            g.setColour(juce::Colours::black);
            g.fillRect(bounds);
            
            g.setGradientFill(gradient);
            const auto scaledY = jmap(level, minRange, maxRange, 0.0f, static_cast<float>(getHeight()));
            g.fillRect(bounds.removeFromBottom(scaledY));
            
            // draw zero clip region
            bounds = getLocalBounds().toFloat();
            g.setColour(juce::Colours::red);
            const auto zeroLevel = jmap(0.0f, minRange, maxRange, 0.0f, static_cast<float>(getHeight()));
            g.drawLine(bounds.getTopLeft().getX(),
                       bounds.getHeight() - zeroLevel,
                       bounds.getTopRight().getX(),
                       bounds.getHeight() - zeroLevel,
                       4); // thickness
        }
        
        void resized() override{
            const auto bounds = getLocalBounds().toFloat();
            const auto zeroLevel = jmap(0.0f, minRange, maxRange, 0.0f, static_cast<float>(getHeight()));
            auto topleft =  bounds.getTopLeft();
            topleft.setY(bounds.getHeight() - zeroLevel);
            gradient = ColourGradient{
                Colours::green,
                bounds.getBottomLeft(),
                Colours::red,
                topleft,
                false
            };
            gradient.addColour(0.8f, juce::Colours::yellow);
        }
            
        void setLevel(const float value){
            level = value;
        }
        
    private:
        float level = -100.0f;
        ColourGradient gradient{};
        float minRange = -100.0f;
        float maxRange = 0.0f;
    };
}
