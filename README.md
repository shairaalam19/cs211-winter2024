# CS211 Winter 2024 Project 3
## Expectations 
### Context
In IoT applications, edge devices are typically constrained in computational capability and limited system support (programming and libraries). Thus running AI models with heavy computation is a challenging task on edge devices. AI accelerators, such as Coral TPU from Google, have been designed to address this issue.
### Problem Statement
In this project, we are going to develop machine learning apps to be applied on edge systems using the Coral TPU accelerator. Despite its Tops-level throughput, it only supports a subset of TensorFlow Lite operations with 8-bit fixed-point precision. Effort is required to produce functional applications with the accelerator. (Details: https://coral.ai/docs/edgetpu/models-intro)
### Goal
The goal of the project is to export a common machine learning application to the edge device with the Coral TPU accelerator and address the challenges in the pipeline, including machine learning model, quantization and accuracy.
### Tasks
1. For a given model (in Tensorflow), convert the TFLite model to be used with the
Coral TPU accelerator. (High priority)
2. In the above process, certain manipulations on the model may be necessary.
Create automated tools to perform the operations programmatically. (Medium
priority)
3. Evaluate the accuracy of the converted (quantized) model. And use quantize-aware training or transfer learning to optimize the model. (High priority) Expected outcome: Optimized models for the edge AI using Coral TPU and a set of helper programs for model conversion and/or optimization.
4. Desired skills
- Tensorflow, Tensorflow Lite
- Python programming

Mentor: Boyan Ding <dboyan@cs.ucla.edu>

## Environment setup
### Python setup
[Link to tflite documentation](https://coral.ai/docs/edgetpu/tflite-python/) (a system with python 3.9 is needed, you can use conda to create a python 3.9 environment)

### Model compiler
[Link to Coral AI Compiler](https://coral.ai/docs/edgetpu/compiler/)

## DLC codebase
[Link to DLC Codebase](DLC)

## Model 
[Link to Model](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut/blob/fcceb7af93d1271633a7d0025a21498cf19863d0/DLC_ma_superquadruped_resnet_50_iteration-0_shuffle-1.tar.gz)

## Visualization script
[Link to Visualization script](import_pb.py)
import_pb.py in this repository 
Use the .pb file in the model as input and the model can be visualized with tensorboard as described in the comments (available after installing tensorflow)

## Tensorflow Model Basics Documentation
[Link to Documentation]('Tensorflow Model Basics.pdf')

## TPU Starter
[Link to TPU Starter](tpu-starter.tar.gz) 

## Tf.graph documentation 
[Link to tf.graph documentation](https://www.tensorflow.org/api_docs/python/tf/Graph#get_operations)

## To Do
1. Produce splitting 

2. First model, compile into tpu format 
