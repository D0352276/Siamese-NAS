import tensorflow as tf
import numpy as np
from graph_conv import GConvGroup

swish=tf.keras.layers.Lambda(lambda x:x*tf.math.sigmoid(x))
hard_sigmoid=tf.keras.layers.Lambda(lambda x:tf.nn.relu6(x+3.0)/6.0)
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))

class ConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=mish,
                 name="convbn"):
        super(ConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class DepthConvBN(tf.Module):
    def __init__(self,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=mish,
                 name="depthconvbn"):
        super(DepthConvBN,self).__init__(name=name)
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding=self._padding,
                                                        use_bias=self._bias,
                                                        name=self._name+"_depthconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._depthconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class SelfAttention(tf.Module):
    def __init__(self,name="selfatten"):
        super(SelfAttention,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._convq=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=mish,name=self._name+"_convq")
        self._convk=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=mish,name=self._name+"_convk")
        self._convv=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=mish,name=self._name+"_convv")
        self._conv_confd=ConvBN(input_ch,kernel_size=(1,1),use_bn=False,activation=hard_sigmoid,name=self._name+"_conv2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._gap(input_ts)
        x=tf.reshape(x,[-1,1,1,input_ch])
        q=self._convq(x)
        k=self._convk(x)
        v=self._convv(x)
        q=tf.reshape(q,[-1,input_ch//4,1])
        k=tf.reshape(k,[-1,1,input_ch//4])
        v=tf.reshape(v,[-1,input_ch//4,1])
        qk=tf.matmul(q,k)
        qkv=tf.matmul(tf.nn.softmax(qk),v)
        qkv=tf.reshape(qkv,[-1,1,1,input_ch//4])
        x=self._conv_confd(qkv)
        output_ts=input_ts*x
        return output_ts

class NodesAttention(tf.Module):
    def __init__(self,name="nodesatten"):
        super(NodesAttention,self).__init__(name=name)
        self._name=name
        self._bulid_bool=False
    @tf.Module.with_name_scope
    def _Build(self,feats_len):
        self._gconvq=GConvGroup(feats_len,name=self._name+"_gconvq")
        self._gconvk=GConvGroup(feats_len,name=self._name+"_gconvk")
        self._gconvv=GConvGroup(feats_len,name=self._name+"_gconvv")
        self._gconvout=GConvGroup(feats_len,name=self._name+"_gconvout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,feats=input_ts
        _,nds_len,feats_len=feats.get_shape().as_list()
        if(self._bulid_bool==False):
            self._bulid_bool=True
            self._Build(feats_len)
        q=self._gconvq([adj,feats])
        k=self._gconvk([adj,feats])
        v=self._gconvv([adj,feats])
        k=tf.transpose(k,[0,2,1])
        qk=tf.matmul(q,k)
        qk_1=qk*adj
        qk_2=qk*tf.transpose(adj,[0,2,1])
        qk_1=tf.nn.softmax(qk_1)
        qk_2=tf.nn.softmax(qk_2)
        qkv_1=tf.matmul(qk_1,v)
        qkv_2=tf.matmul(qk_2,v)
        output_ts=feats+qkv_1+qkv_2
        output_ts=self._gconvout([adj,output_ts])
        return output_ts

class ExpandBlock(tf.Module):
    def __init__(self,t=3,activation=mish,name="expblck"):
        super(ExpandBlock,self).__init__(name=name)
        self._t=t
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._dwconv_list=[]
        for i in range(self._t):
            dwconv=DepthConvBN((3,3),activation=self._activation,name=self._name+"_dwconv_"+str(i))
            self._dwconv_list.append(dwconv)
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        exp_ts_list=[]
        x=input_ts
        for i in range(self._t):
            x=self._dwconv_list[i](x)+input_ts
            exp_ts_list.append(x)
        out_ts=tf.concat(exp_ts_list,axis=-1)
        return out_ts

class Resize(tf.Module):
    def __init__(self,output_hw,name="resize"):
        super(Resize,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class Split(tf.Module):
    def __init__(self,split_parts=2,name="split"):
        super(Split,self).__init__(name=name)
        self._split_parts=split_parts
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        res=input_ch%self._split_parts
        self._part_ch=int((input_ch+res)/self._split_parts)
        if(res>0):
            self._rechannel=ConvBN(input_ch+res,(1,1),activation=False,name=self._name+"_rechannel")
        else:
            self._rechannel=None
        return
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        output_ts_list=[]
        if(self._rechannel!=None):
            input_ts=self._rechannel(input_ts)
        for i in range(self._split_parts):
            output_ts=input_ts[...,int(self._part_ch*i):int(self._part_ch*(i+1))]
            output_ts_list.append(output_ts)
        return output_ts_list

class Identity(tf.Module):
    def __init__(self,name="identity"):
        super(Identity,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.identity(input_ts)
        return output_ts

class ReLUConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=tf.nn.relu,
                 name="convbn"):
        super(ReLUConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._act(input_ts)
        x=self._conv(x)
        if(self._use_bn==True):x=self._bn(x)
        output_ts=x
        return output_ts

class Split2(tf.Module):
    def __init__(self,name="split2"):
        super(Split2,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts_list=[input_ts,input_ts]
        return output_ts_list

class Split3(tf.Module):
    def __init__(self,name="split3"):
        super(Split3,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts_list=[input_ts,input_ts,input_ts]
        return output_ts_list

class Zeros(tf.Module):
    def __init__(self,name="zeros"):
        super(Zeros,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.zeros(tf.shape(input_ts))
        return output_ts