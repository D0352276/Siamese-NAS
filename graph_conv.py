import tensorflow as tf
from tensorflow.keras import activations,regularizers,constraints,initializers
spdot=tf.sparse.sparse_dense_matmul
dot=tf.matmul
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))

class GCNConv(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(GCNConv,self).__init__()
        self.units=units
        self.activation=activations.get(activation)
        self.use_bias=use_bias
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.bias_initializer=initializers.get(bias_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        self.bias_regularizer=regularizers.get(bias_regularizer)
        self.activity_regularizer=regularizers.get(activity_regularizer)
        self.kernel_constraint=constraints.get(kernel_constraint)
        self.bias_constraint=constraints.get(bias_constraint)
    def build(self,input_shape):
        fdim=input_shape[1][2]
        if not hasattr(self,"weight"):
            self.weight=self.add_weight(name="weight",
                                          shape=(fdim,self.units),
                                          initializer=self.kernel_initializer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self,"bias"):
                self.bias=self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GCNConv,self).build(input_shape)
    def call(self,inputs):
        self.An=inputs[0]
        self.X=inputs[1]
        if isinstance(self.X,tf.SparseTensor):
            h=spdot(self.X,self.weight)
        else:
            h=dot(self.X,self.weight)
        output=dot(self.An,h)
        if self.use_bias:
            output=tf.nn.bias_add(output,self.bias)
        if self.activation:
            output=self.activation(output)
        return output

class GConvGroup(tf.Module):
    def __init__(self,filters,activation=mish,use_bn=True,name="gconvgroup"):
        super(GConvGroup,self).__init__(name=name)
        self._filters=filters
        self._activation=activation
        self._use_bn=use_bn
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._gcnconv_1=GCNConv(self._filters,activation=None,name=self._name+"_gcnconv_1")
        if(self._use_bn==True):self._bn1=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn1")
        self._gcnconv_2=GCNConv(self._filters,activation=None,name=self._name+"_gcnconv_2")
        if(self._use_bn==True):self._bn2=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn2")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,x=input_ts
        x1=self._gcnconv_1([adj,x])
        if(self._use_bn==True):x1=self._bn1(x1)
        x1=self._act(x1)
        x2=self._gcnconv_2([tf.transpose(adj,[0,2,1]),x])
        if(self._use_bn==True):x2=self._bn2(x2)
        x2=self._act(x2)
        out_ts=(x1+x2)/2
        return out_ts