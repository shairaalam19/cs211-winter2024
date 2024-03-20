import argparse
import tensorflow as tf
import numpy as np
import cv2
import sys

VIDEO_FILE = './rhino_7.mp4'
INPUT_MODEL_PATH = './DLC_ma_sub_p1_320_320'
INPUT_DATA = []

def load_video():
    cap = cv2.VideoCapture(VIDEO_FILE)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    dataset = []

    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 320))
        # frame_u = skimage.img_as_ubyte(frame)
        dataset.append(np.expand_dims(frame, axis=0).astype(np.float32))

    return dataset

def make_rep_dataset(dataset):
    def func():
        for item in dataset:
            yield [item]

    return func

def main():
    parser = argparse.ArgumentParser(
        prog='dlc_convert.py',
        description='Convert DLC model into TFlite'
    )
    parser.add_argument('-m', '--model', type=str, default=INPUT_MODEL_PATH,
                        help='Path of input model (TF1 SavedModel)')
    parser.add_argument('-O', '--opt', default='none',
                        choices=['none', 'drange', 'float16', 'int_fallback', 'int8_pure'],
                        help='Optimization type, the last two requires -r argument')
    parser.add_argument('-r', '--rep', help='Use video as representative data')
    parser.add_argument('-w', '--width', type=int, default=320, help='Width of video')
    parser.add_argument('-t', '--height', type=int, default=320, help='Height of video')
    parser.add_argument('-o', '--output', type=str, default='output.tflite')

    args = parser.parse_args()

    if args.opt in ['int_fallback', 'int8_pure'] and args.rep is None:
        print('Error: Representative dataset should be given for full int quantization')
        return

    if args.opt == 'drange' and args.rep is not None:
        print('Error: drange optimization should not come with representative dataset')
        return

    if args.rep is not None:
        print('Loading input video as typ data')
        dataset = load_video(args.rep)
        print(f'{len(dataset)} frames loaded')

    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)

    converter = tf.lite.TFLiteConverter.from_saved_model(args.model)

    # https://www.tensorflow.org/lite/performance/post_training_quantization
    if args.opt != 'none':
        # The only operation in drange
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.opt == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        elif args.opt in ['int_fallback', 'int8_pure']:
            converter.representative_dataset = make_rep_dataset(dataset)
            if args.opt == 'int8_full':
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(args.output, 'wb') as f:
        f.write(tflite_quant_model)

if __name__ == '__main__':
    main()
