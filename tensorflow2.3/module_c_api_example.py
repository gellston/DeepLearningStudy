import numpy as np
import tensorflow as tf
import pydotplus
import cv2

from model.ModuleExample import ModuleExample
from tensorflow.keras.models import model_from_json
from util.segmentation_dataloader import segmentation_dataloader

print(tf.__version__)


def inspect_checkpoint(checkpoint, print_values=False, variables=None):
    if not variables:
        variables = [var_name for (var_name, shape) in tf.train.list_variables(checkpoint)]

    checkpoint_reader = tf.train.load_checkpoint(checkpoint)
    for var_name in variables:

        try:
            tensor = checkpoint_reader.get_tensor(var_name)
        except Exception as e:
            print('ignored   : %s (exception %s)' % (var_name, str(type(e))))
            continue
        if isinstance(tensor, np.ndarray):
            if print_values:
                print('tensor    : ', var_name, tensor.shape, tensor)
            else:
                print('tensor    : ', var_name, tensor.shape)
        else:
            if print_values:
                print('non-tensor: ', var_name, type(tensor), tensor)
            else:
                print('non-tensor: ', var_name, type(tensor))

tf.config.run_functions_eagerly(False)

module = ModuleExample()

inspect_checkpoint("C:\\Github\\DeepLearningStudy\\trained_model\\ModuleExample/variables/variables")

signatures = {'prediction':module.__call__.get_concrete_function(tf.TensorSpec([None, 256, 256, 3], tf.float32)),
              'accuracy':module.accracy_on_batch.get_concrete_function(tf.TensorSpec([None, 256, 256, 3], tf.float32),tf.TensorSpec([None, 256, 256, 1], tf.float32)),
              'train':module.train_one_batch.get_concrete_function(tf.TensorSpec([None, 256, 256, 3], tf.float32),tf.TensorSpec([None, 256, 256, 1], tf.float32))}

tf.saved_model.save(module, "C:\\Github\\DeepLearningStudy\\trained_model\\ModuleExample", signatures=signatures)



