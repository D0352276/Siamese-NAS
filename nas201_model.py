import tensorflow as tf
import numpy as np
from modules import ReLUConvBN,Split2,Split3,Identity,Zeros

class Nas201Cell(tf.Module):
    def __init__(self,filters,ops,adj,name="nas201cell"):
        super(Nas201Cell,self).__init__(name=name)
        self._filters=filters
        self._adj=np.array(adj)
        self._max_nodes=np.shape(adj)[0]
        self._ops=ops
        self._name=name
        self._Build(ops)
    @tf.Module.with_name_scope
    def _Build(self,ops):
        self._ts_ops=[]
        op=None
        for i,chosen_op in enumerate(ops):
            if(chosen_op=="split_3"):
                op=Split3(name=self._name+"_op_"+str(i))
            elif(chosen_op=="split_2"):
                op=Split2(name=self._name+"_op_"+str(i))
            elif(chosen_op=="none"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(chosen_op=="skip_connect"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(chosen_op=="nor_conv_1x1"):
                op=ReLUConvBN(self._filters,(1,1),name=self._name+"_op_"+str(i))
            elif(chosen_op=="nor_conv_3x3"):
                op=ReLUConvBN(self._filters,(3,3),name=self._name+"_op_"+str(i))
            elif(chosen_op=="zeros"):
                op=Zeros(name=self._name+"_op"+str(i))
            elif(chosen_op=="avg_pool_3x3"):
                op=tf.keras.layers.AveragePooling2D((3,3),strides=(1,1),padding="same",name=self._name+"_op_"+str(i))
            elif(chosen_op=="output"):
                op=Identity(name=self._name+"_op_"+str(i))
            else:
                op=Identity(name=self._name+"_op_"+str(i))
            self._ts_ops.append(op)
        return
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        cur_ts_list=[[] for i in range(self._max_nodes)]
        cur_ts_list[0].append(input_ts)
        for i,chosen_op in enumerate(self._ops):
            if(len(cur_ts_list[i])>1):
                shapes=tf.zeros([1,3])
                shapes=tf.cast(shapes,tf.int32)
                for cur_ts in cur_ts_list[i]:
                    cur_shape=tf.reshape(tf.shape(cur_ts)[1:],[1,3])
                    shapes=tf.concat([shapes,cur_shape],axis=0)
                shapes=tf.cast(shapes,tf.float32)
                max_h=tf.reduce_max(shapes[:,0])
                max_w=tf.reduce_max(shapes[:,1])
                for j,cur_ts in enumerate(cur_ts_list[i]):
                    cur_ts_list[i][j]=tf.image.resize(cur_ts,[max_h,max_w],method=tf.image.ResizeMethod.BILINEAR)
                x=tf.keras.layers.Add()(cur_ts_list[i])
            else:
                x=cur_ts_list[i][0]
            if(chosen_op=="output"):
                output_ts=x
            elif(chosen_op=="split_2" or chosen_op=="split_3"):
                target_ts_list=self._ts_ops[i](x)
                target_idxes=np.where(self._adj[i]==1)[0]
                for j,target_ts in enumerate(target_ts_list):
                    target_idx=target_idxes[j]
                    cur_ts_list[target_idx].append(target_ts)
            else:
                target_ts=self._ts_ops[i](x)
                target_idx=np.where(self._adj[i]==1)[0][0]
                cur_ts_list[target_idx].append(target_ts)
        in_ch=input_ts.get_shape().as_list()[3]
        out_ch=output_ts.get_shape().as_list()[3]
        if(in_ch==out_ch):
            output_ts=input_ts+output_ts
        return output_ts

class Nas201Block(tf.Module):
    def __init__(self,filters,ops,adj,blck_len=1,name="nas201block"):
        super(Nas201Block,self).__init__(name=name)
        self._filters=filters
        self._blck_len=blck_len
        self._adj=adj
        self._ops=ops
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._cell_list=[]
        self._first_cell=Nas201Cell(self._filters,self._ops,self._adj,name=self._name+"_first_cell")
        for i in range(self._blck_len-1):
            self._cell_list.append(Nas201Cell(self._filters,self._ops,self._adj,name=self._name+"_cell_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        first_x=self._first_cell(input_ts)
        x=first_x
        for i in range(self._blck_len-1):
            x=self._cell_list[i](x)
        if(self._blck_len>1):
            output_ts=first_x+x
        else:
            output_ts=first_x
        return output_ts

class ResBlock(tf.Module):
    def __init__(self,filters,name="resblock"):
        super(ResBlock,self).__init__(name=name)
        self._filters=filters
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv_1=ReLUConvBN(self._filters,(3,3),(2,2),name=self._name+"_conv_1")
        self._conv_2=ReLUConvBN(self._filters,(3,3),(1,1),name=self._name+"_conv_2")
        self._pooling=tf.keras.layers.AveragePooling2D((2,2),name=self._name+"_pooling")
        self._conv_3=ReLUConvBN(self._filters,(1,1),(1,1),use_bn=False,activation=None,name=self._name+"_conv_3")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv_1(input_ts)
        x=self._conv_2(x)
        shortcut=self._pooling(input_ts)
        shortcut=self._conv_3(shortcut)
        output_ts=x+shortcut
        return output_ts

class Nas201Model(tf.Module):
    def __init__(self,ops,adj,init_channel=16,blck_len=5,name="nas201model"):
        super(Nas201Model,self).__init__(name=name)
        self._ops=ops
        self._adj=adj
        self._init_channel=init_channel
        self._blck_len=blck_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._convbn=ReLUConvBN(self._init_channel,(3,3),(1,1),activation=None,name=self._name+"_convbn")
        self._cells_1=Nas201Block(self._init_channel,self._ops,self._adj,self._blck_len,name=self._name+"_cells_1")
        self._resblck_1=ResBlock(int(self._init_channel*2),name=self._name+"_resblck_1")
        self._cells_2=Nas201Block(int(self._init_channel*2),self._ops,self._adj,self._blck_len,name=self._name+"_cells_2")
        self._resblck_2=ResBlock(int(self._init_channel*4),name=self._name+"_resblck_2")
        self._cells_3=Nas201Block(int(self._init_channel*4),self._ops,self._adj,self._blck_len,name=self._name+"_cells_3")
        self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(tf.nn.relu,name=self._name+"_act")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._convbn(input_ts)
        x=self._cells_1(x)
        x=self._resblck_1(x)
        x=self._cells_2(x)
        x=self._resblck_2(x)
        x=self._cells_3(x)
        x=self._bn(x)
        x=self._act(x)
        out_ts=self._gap(x)
        return out_ts