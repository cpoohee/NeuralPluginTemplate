# NeuralPluginTemplate
An audio plugin template based on JUCE to load an ONNX AI model.
- it uses the dynamic lib of ONNX runtime compliled for MACOS. 
- the current Projucer is aimed to deploy AU and VST3 for MAC OS only. 

The current controls include the input/output gain and mix level, and also a phase flip check box. 

There are 2 basic level meter for input and output.

# Quick instructions
- place the ONNX Model under the folder `./model`
- at the constructor of `PluginProcessor.cpp`
  - edit the file path e.g : `auto model_file = bundle.getChildFile ("Resources/model/waveunet_distort.onnx");`
  - and also it's intended blocksize
  
- Modify `OnnxModel.cpp` 's process() for model's input/outputs processing.

- the current onnxmodel is a noise producing model sample.

- this repo contains a version 1.14.1 ONNX runtime library for MACOS, which the projucer is configured to run RPath with it.

# Notes
- the process is currently mixed down to mono for the ML model's input. 
- Do modify both `OnnxModel.cpp` and `PluginProcessor.cpp` if your model needs stereo input.

# Related
- See https://github.com/cpoohee/MLPluginTemplate for the Machine learning audio template.
