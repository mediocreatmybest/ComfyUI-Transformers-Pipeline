# ComfyUI Transformers Pipeline

## Overview
A collection of additional ComfyUI nodes for utilizing the Hugging Face Transformers pipeline in a modular and efficient manner.

## Installation

Can be installed through the ComfyUI manager or ComfyUI cli

## Caption your first folder

Open [Caption My Images](example_workflows/Caption_My_Images.json) in ComfyUI Desktop. Enter your image folder in **Choose your image folder**, then click **Run**. The first run may download the BLIP model. Preview shows the images with their captions. To see another page, change the Preview offset and run again.

To write text files beside your images, open the separate [Simple export example](example_workflows/Simple.json).

## Nodes

### **Primary Nodes**
- **Model Loader**: Loads transformer models.
- **Caption Generator**: Generates image captions.
- **Batch Processor**: Simple node to specify and handle lists to batch processing.
- **Export Captions**: Simple node to save text output to files based on original file input names.

### **Utility Nodes**
- **Debug Node**: Allows connecting list nodes to a text output to view node outputs
- **Debug Model Node**: Allows connecting the model loader node to a text output to view node outputs and check configuration
- **EXIF Metadata Extractor**: Extracts camera metadata from images.
- **Task List Node**: Provides preset transformer tasks.

## TODO

### **Short Term**
- [ ] Add tooltips nodes.
- [ ] Move pipeline nodes into its own folders (that’s our primary focus really) with non-pipeline tasks separated.
- [ ] fix logging to possibly Python’s logging module. 
- [ ] Add additional example nodes

### **Long Term**

- [x] Register in ComfyUI registry
- [ ] look into multi-pass captions or outputs
- [ ] add additional pipeline tasks
- [ ] add multimodal tasks to the mix
- [ ] Add additional output nodes
- [ ] Allow setting different file extensions for output

## Contributing
Pull requests are welcome.
As well as Ideas, suggestions, anything at all. 
What works, what doesn’t, etc.

## License
Please see license file. 

