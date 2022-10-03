import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def FLOPs(model):
    forward_pass=tf.function(model.call,input_signature=[tf.TensorSpec(shape=(1,)+model.input_shape[1:])])
    graph_info=profile(forward_pass.get_concrete_function().graph,options=ProfileOptionBuilder.float_operation())
    flops=graph_info.total_float_ops//2
    print('Flops:{:,}'.format(flops))
    return flops