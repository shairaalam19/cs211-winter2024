import os
import pathlib
import time
import cv2
import numpy as np
import tensorflow as tf
from pycoral.utils import edgetpu

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
MODEL_FILE = os.path.join(SCRIPT_DIR, 'output640_360_edgetpu.tflite')
VIDEO_FILE = './test_video-selected/rhino_5.mp4'

def main():
    print('Reading input')
    cap = cv2.VideoCapture(VIDEO_FILE)
    ret, frame = cap.read()

    if not ret:
        print("Error, failed to read from input")
        return

    frame = cv2.resize(frame, (640, 360))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Load the TensorFlow model
    tf1 = tf.compat.v1
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile("snapshot-700000.pb", "rb") as f: gdef.ParseFromString(f.read())
    #with tf1.io.gfile.GFile("saved_model.pb", "rb") as f: gdef.ParseFromString(f.read())


    with tf1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(gdef, name="")
        input = sess.graph.get_tensor_by_name('Placeholder:0')
        output =  sess.graph.get_tensor_by_name('concat_1:0') 
        start_time = time.time()
        tensorflow_output = sess.run(output, feed_dict={input: np.expand_dims(frame, axis=0)})
        inference_time_tf = time.time() - start_time
    

    print(tensorflow_output)
    print("Inference time with TensorFlow model:", inference_time_tf)



if __name__ == '__main__':
    main()