# NeuralPluginTemplate
A audio plugin template based on JUCE to load an ONNX AI model.
- it uses the dynamic lib of ONNX runtime compliled for MACOS. 
- the current Projucer is aimed to deploy for MAC OS only. 

# Quick instructions
- place the ONNX Model under the folder `./model`
- at the constructor of `PluginProcessor.cpp`
  - edit the file path : `auto model_file = bundle.getChildFile ("Resources/model/waveunet_distort.onnx");`
  - and also it's intended blocksize
  
- Modify OnnxModel.cpp 's process() for modeling input/outputs.


# Notes
- the process is currently mixed down to mono as the input to the ML model. DO modify as if your model needs stereo input.

# Related
- See https://github.com/cpoohee/MLPluginTemplate for the Machine learning for audio template
