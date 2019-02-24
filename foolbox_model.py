import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

import foolbox
from foolbox import zoo

def create():
    # load pretrained weights
    weights_path = zoo.fetch_weights(
        'http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz',
        unzip=True
    )
    checkpoint = os.path.join(weights_path, 'imagenet64_alp025_2018_06_26.ckpt')

    # load model
    input_ = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    model_fn_two_args = get_model('resnet_v2_50', 1001)
    model_fn = lambda x: model_fn_two_args(x, is_training=False)
    preprocessed = _normalize(input_)
    logits = model_fn(preprocessed)[:, 1:]

    # load pretrained weights into model
    variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session().__enter__()

    saver.restore(sess, checkpoint)

    # create foolbox model
    fmodel = foolbox.models.TensorFlowModel(input_, logits, bounds=(0, 255), preprocessing=(0, 255))
    
    return fmodel

def _normalize(image):
  """Rescale image to [-1, 1] range."""
  return tf.multiply(tf.subtract(image, 0.5), 2.0)

def get_model(model_name, num_classes):
  """Returns function which creates model.

  Args:
    model_name: Name of the model.
    num_classes: Number of classes.

  Raises:
    ValueError: If model_name is invalid.

  Returns:
    Function, which creates model when called.
  """
  if model_name.startswith('resnet'):
    def resnet_model(images, is_training, reuse=tf.AUTO_REUSE):
      with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_fn = resnet_v2.resnet_v2_50
        logits, _ = resnet_fn(images, num_classes, is_training=is_training,
                              reuse=reuse)
        logits = tf.reshape(logits, [-1, num_classes])
      return logits
    return resnet_model
  else:
    raise ValueError('Invalid model: %s' % model_name)
