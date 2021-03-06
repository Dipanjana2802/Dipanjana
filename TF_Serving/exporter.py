#Reference
#https://gist.github.com/datlife/2c39a1893e689130c9a18ff14ec452a0

from __future__ import print_function

import os
import tensorflow as tf

# Load frozen graph utils
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile

# TF Libraries to export model into .pb file
from tensorflow.python.client import session
from tensorflow.python.saved_model import signature_constants
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.tools.graph_transforms import TransformGraph

# Path to frozen inference graph in your system
frozen_graph = '/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

# Name of the model this should be same with the model name inside the client.py file
model_name     = 'faster_rcnn_inception_v2_coco'

# Output location for the servable model
output_loc       = '/Running_Object_Detection_Model_TF_SERVING/'

def _main_():
    # #################
    # Setup export path
    ###################
    version    = 1
    output_dir = os.path.join(output_loc, model_name)
    export_path = os.path.join(output_dir, str(version))

    # ######################
    #  Interference Pipeline
    # ######################
    input_names = 'image_tensor'
    output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']

    with tf.Session() as sess:
        input_tensor = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name=input_names)
        # ###################
        # load frozen graph
        # ###################
        graph_def = load_graph_from_pb(frozen_graph)
        outputs = tf.import_graph_def(graph_def,
                                      input_map={'image_tensor': input_tensor},
                                      return_elements=output_names,
                                      name='')
        outputs = [sess.graph.get_tensor_by_name(ops.name +':0')for ops in outputs]
        outputs = dict(zip(output_names, outputs))

    # #####################
    # Quantize Frozen Model
    # #####################
    transforms = ["add_default_attributes",
                  "quantize_weights", "round_weights",
                  "fold_batch_norms", "fold_old_batch_norms"]

    quantized_graph = TransformGraph(input_graph_def=graph_def,
                                     inputs=input_names,
                                     outputs=output_names,
                                     transforms=transforms)

    # #####################
    # Export to TF Serving#
    # #####################
    # Reference: https://github.com/tensorflow/models/tree/master/research/object_detection

    with tf.Graph().as_default():
        tf.import_graph_def(quantized_graph, name='')

        # Optimizing graph
        rewrite_options = rewriter_config_pb2.RewriterConfig()
        rewrite_options.optimizers.append('pruning')
        rewrite_options.optimizers.append('constfold')
        rewrite_options.optimizers.append('layout')
        graph_options = tf.GraphOptions(rewrite_options=rewrite_options, infer_shapes=True)

        # Build model for TF Serving
        config = tf.ConfigProto(graph_options=graph_options)

        # @TODO: add XLA for higher performance (AOT for ARM, JIT for x86/GPUs)
        # https://www.tensorflow.org/performance/xla/
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        # Reference:
        # https://www.tensorflow.org/guide/saved_model
        with session.Session(config=config) as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(input_tensor)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs     = tensor_info_inputs,
                            outputs    = tensor_info_outputs,
                            method_name= signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING], #tag_constants.SERVING is IMP to specify as this indicates the saved graph is meant for serving
                    signature_def_map={'predict_images': detection_signature,
                                       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: detection_signature,
                                       },
            )
            builder.save()

    print("\n\nModel is ready for TF Serving. (saved at {}/saved_model.pb)".format(export_path))

def load_graph_from_pb(model_filename):
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(data)
    return graph_def



if __name__ == '__main__':
    _main_()
