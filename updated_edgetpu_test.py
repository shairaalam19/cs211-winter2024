import os
import pathlib
import time
import cv2
import numpy as np
import tensorflow as tf
from pycoral.utils import edgetpu

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
MODEL_FILE = os.path.join(SCRIPT_DIR, 'output640_360_edgetpu.tflite')
VIDEO_FILE = './test_video-selected/rhino_7.mp4'

def main():
    print('Reading input')
    cap = cv2.VideoCapture(VIDEO_FILE)
    ret, frame = cap.read()

    if not ret:
        print("Error, failed to read from input")
        return

    frame = cv2.resize(frame, (640, 360))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print('Making TPU interpreter')
    interpreter = edgetpu.make_interpreter(MODEL_FILE)
    interpreter.allocate_tensors()

    print('Input:')
    print(interpreter.get_input_details())
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('\nOutput:')
    print(interpreter.get_output_details())

    interpreter.set_tensor(input_details[0]['index'], np.float32(np.expand_dims(frame, axis=0)))
    interpreter.invoke()
    res1 = interpreter.get_tensor(output_details[0]['index'])
    res2 = interpreter.get_tensor(output_details[1]['index'])

    print('int1:')
    print(res1)
    print(output_details[0]['index'])
    print('int2:')
    print(res2)
    print(output_details[1]['index'])

    # Load the TensorFlow model
    tf1 = tf.compat.v1
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile("snapshot-700000.pb", "rb") as f: gdef.ParseFromString(f.read())
    #with tf1.io.gfile.GFile("saved_model.pb", "rb") as f: gdef.ParseFromString(f.read())


    with tf1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(gdef, name="")
        input_tensor1_tensorflow = sess.graph.get_tensor_by_name('pose/part_pred/block4/BiasAdd:0')
        input_tensor2_tensorflow = sess.graph.get_tensor_by_name('pose/locref_pred/block4/BiasAdd:0') 
        outputs =  sess.graph.get_tensor_by_name('concat_1:0') 
        start_time = time.time()
        tensorflow_output = sess.run(outputs, feed_dict={input_tensor1_tensorflow: res1, input_tensor2_tensorflow: res2})
        inference_time_tf = time.time() - start_time

    print(tensorflow_output)
    print("Total Inference time:", inference_time_tf)


    for i in range(10):
        start = time.time()
        interpreter.invoke()
        end = time.time()
        print(f'Iter {i}: T={end-start}')

if __name__ == '__main__':
    main()