# Tensorflow-Object-Detection-Inference
This is minimal script to run inference on the test images and dump JSON results with detected classes for each image

## Directory Structure:
```console
|-- Object_Detection_Inference_Minimal_V1.py
|-- output.json
|-- README.md
|-- LICENSE
|-- object_detection
    |-- Inference_Graph
    |-- Label_Map
        |-- labelmap.pbtxt
    |-- Test_Images
        |-- dog.jpg
        |-- cat.jpg
    |-- utils
        |-- label_map_util.py
        |-- ops.py       

```
## Setup Steps:
- Copy exported inference graph files and folders into object_detection -> Inference_Graph directory.
- Copy your labelmap.pbtxt into object_detection -> Label_Map directory.
- Copy test images into object_detection -> Test_Images directory.
- Run the inference script 
```console
foo@bar:~$ python Object_Detection_Inference_Minimal_V1.py
```
- The output is dumped into the top directory with a name output.json
