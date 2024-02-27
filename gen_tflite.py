import tensorflow as tf
tf1 = tf.compat.v1

ORIG_MODEL="snapshot-1000.pb"
OUTPUT_MODEL="DLC_ma_sub_p1_320_320"

def main():
    gdef = tf1.GraphDef()
    with tf1.io.gfile.GFile(ORIG_MODEL,"rb") as f:
        gdef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
       tf.graph_util.import_graph_def(gdef, name='')

    new_graph=tf.Graph()
    with new_graph.as_default():
        new_input = tf1.placeholder(dtype=tf.float32, shape=[1,320,320,3], name='Placeholder')
        tf1.import_graph_def(g.as_graph_def(), name='', input_map={'Placeholder':new_input})

    gdef_sub = tf1.graph_util.extract_sub_graph(
        new_graph.as_graph_def(),
        ['pose/part_pred/block4/BiasAdd',
         'pose/locref_pred/block4/BiasAdd'])

    g2 = tf.Graph()
    with g2.as_default():
        tf.graph_util.import_graph_def(gdef_sub, name='')

    g2_input = g2.get_tensor_by_name('Placeholder:0')
    g2_part_pred_output = g2.get_tensor_by_name('pose/part_pred/block4/BiasAdd:0')
    g2_locref_pred_output = g2.get_tensor_by_name('pose/locref_pred/block4/BiasAdd:0')

    with tf1.Session(graph=g2) as s2:
        tf1.saved_model.simple_save(session=s2,
            export_dir=OUTPUT_MODEL,
            inputs={'input':g2_input},
            outputs={
                'part_pred_output': g2_part_pred_output,
                'locref_pred_output': g2_locref_pred_output})

if __name__ == '__main__':
    main()
