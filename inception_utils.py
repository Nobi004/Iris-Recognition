from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contrib
import tensorflow as tf



def inception_arg_scope(weight_decay = 0.00004,
                        use_batch_norm = True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon= 0.001,
                        activation_fn = tf.nn.relu,
                        batch_norm_updates_collections= tf.GraphKeys.UPDATE_OPS):
    """Defines the default arg scope for inception models.
    Arg:
     weight_decay:The weight decay to use for regularizing the model.
     use_batch_norm:"IF 'True' batch_norm is applied after each convolution.
     batch_norm_decay: Decay for batch norm moving average.
     batch_norm_epsilon: Small float added to varience to avoid dividing by zero
       in batch norm.
    activation_fn : Activation function for conv2D.
    batch_norm_updates_collections: Collection for the update ops for
       batch norm.
    Returns:
        An 'arg_scope' to use for the inception models.
    """

    batch_norm_params = {
              # Decay for the moving averages.
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,
    }

    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    with slim.arg_scope(
        [tf.keras.layers.Conv2D],
        weights_initializer= slim.varience_scaling_initializer(),
        activation_fn = activation_fn,
        normalizer_fn = normalizer_fn,
        normalizer_params = normalizer_params) as sc:
        return sc
