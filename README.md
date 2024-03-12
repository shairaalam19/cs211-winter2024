# CS211 Winter 2024 Project 3
## Expectations 
### Context
In IoT applications, edge devices are typically constrained in computational capability and limited system support (programming and libraries). Thus running AI models with heavy computation is a challenging task on edge devices. AI accelerators, such as Coral TPU from Google, have been designed to address this issue.
### Problem Statement
In this project, we are going to develop machine learning apps to be applied on edge systems using the Coral TPU accelerator. Despite its Tops-level throughput, it only supports a subset of TensorFlow Lite operations with 8-bit fixed-point precision. Effort is required to produce functional applications with the accelerator. (Details: https://coral.ai/docs/edgetpu/models-intro)
### Goal
The goal of the project is to export a common machine learning application to the edge device with the Coral TPU accelerator and address the challenges in the pipeline, including machine learning model, quantization and accuracy.
### Tasks
1. For a given model (in Tensorflow), convert the TFLite model to be used with the Coral TPU accelerator. (High priority)
2. In the above process, certain manipulations on the model may be necessary. Create automated tools to perform the operations programmatically. (Medium priority)
3. Evaluate the accuracy of the converted (quantized) model. And use quantize-aware training or transfer learning to optimize the model. (High priority) Expected outcome: Optimized models for the edge AI using Coral TPU and a set of helper programs for model conversion and/or optimization.
4. Desired skills

   (a) Tensorflow, Tensorflow Lite
   
   (b) Python programming

Mentor: Boyan Ding <dboyan@cs.ucla.edu>

## References & Resources
### Python setup
[Link to tflite documentation](https://coral.ai/docs/edgetpu/tflite-python/) (a system with python 3.9 is needed, you can use conda to create a python 3.9 environment)

### Model compiler
[Link to Coral AI Compiler](https://coral.ai/docs/edgetpu/compiler/)

### DLC codebase
[Link to DLC Codebase](DLC)

### Model 
[Link to Model](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut/blob/fcceb7af93d1271633a7d0025a21498cf19863d0/DLC_ma_superquadruped_resnet_50_iteration-0_shuffle-1.tar.gz)

### Visualization script
[Link to Visualization script](import_pb.py)
import_pb.py in this repository 
Use the .pb file in the model as input and the model can be visualized with tensorboard as described in the comments (available after installing tensorflow)

### Tensorflow Model Basics Documentation
[Link to Documentation](<Tensorflow Model Basics.pdf>)

### TPU Starter
[Link to TPU Starter](tpu-starter.tar.gz) 

### Tf.graph documentation 
[Link to tf.graph documentation](https://www.tensorflow.org/api_docs/python/tf/Graph#get_operations)

## Steps to Model 
1. run the gen_tflite.py script. This file is based on the gen_lite_model.py in the tpu-starter. This will create a new directory /DLC_ma_sub_p1_320_320 which will contain the model that needs to be trained.
   a. Ensure this file is in the same directory level as the snapshot-1000.pb file
```bash
python gen_tflite.py
```
2. run the train.py script. This file is based on the dlc_convert.py in the tpu-starter. This will train the compressed model and convert it to a tflite model. This will generate a file named output.tflite which will be compiled into a file that's compatible with the Edge TPU.
   a. Ensure this file is in the same directory level as the /DLC_ma_sub_p1_320_320 directory
```bash
python train.py
```
## Steps to Compile 
1. Compile the tflite model using the following documentation: [Link to Coral AI Compiler](https://coral.ai/docs/edgetpu/compiler/)
```bash
edgetpu_compiler output.tflite
```
2. Run import_pb.py by running this command: 
```bash
python import_pb.py --graph=<pb-filename>.pb  --log_dir=./tb_logs
```
In this instance, we use snapshot-1000.pb
```bash
python import_pb.py --graph=<pb-filename>.pb  --log_dir=./tb_logs
```
3. Get the graph on localhost 
```bash
tensorboard --logdir=tb_logs --port=6006 --host=localhost
```
## Steps to Split Graph
Refer to [documentation](#tensorflow-model-basics-documentation)
1. Double click on the "Import" box 
2. Loading model is in 

## Run Model on TPU
Follow these steps to get started with the Edge TPU: [Link to Coral Edge TPU Instructions](https://coral.ai/docs/accelerator/get-started/#requirements)
1. 

## Model from TPU to non-TPU 
1. Get output of model on TPU 
2. Use the output of that model as an input on the model that is not on the TPU 

## Items to do 
1. [Model](#steps-to-model) 
2. [Compile into TPU format](#steps-to-compile) 
3. [Split graph](#steps-to-split-graph)
4. [Run Model on TPU](#run-model-on-tpu)
5. [Model from TPU to non-TPU](#model-from-tpu-to-non-tpu)
