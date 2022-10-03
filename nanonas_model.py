import tensorflow as tf
import numpy as np
from modules import ConvBN
from modules import DepthConvBN,ConvBN,SelfAttention,ExpandBlock,Split,Identity

class NanoNasCell(tf.Module):
    def __init__(self,filters,ops,adj,name="nanonascell"):
        super(NanoNasCell,self).__init__(name=name)
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
            if(chosen_op=="none"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(chosen_op=="dwconv3x3"):
                op=DepthConvBN((3,3),name=self._name+"_op_"+str(i))
            elif(chosen_op=="conv1x1"):
                op=ConvBN(self._filters,(1,1),name=self._name+"_op_"+str(i))
            elif(chosen_op=="atten"):
                op=SelfAttention(name=self._name+"_op"+str(i))
            elif(chosen_op=="expand"):
                op=ExpandBlock(name=self._name+"_op"+str(i))
            elif(chosen_op=="split"):
                op=Split(name=self._name+"_op"+str(i))
            elif(chosen_op=="output"):
                op=ConvBN(self._filters,(1,1),activation=None,name=self._name+"_op_"+str(i))
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
                x=tf.concat(cur_ts_list[i],axis=-1)
            else:
                x=cur_ts_list[i][0]
            if(chosen_op=="output"):
                out_ch=x.get_shape().as_list()[3]
                if(out_ch==self._filters):
                    output_ts=x
                else:   
                    output_ts=self._ts_ops[i](x)
            elif(chosen_op=="split"):
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

class NanoNasBlock(tf.Module):
    def __init__(self,filters,ops,adj,blck_len=1,name="nanonasblock"):
        super(NanoNasBlock,self).__init__(name=name)
        self._filters=filters
        self._blck_len=blck_len
        self._adj=adj
        self._ops=ops
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._cell_list=[]
        self._first_cell=NanoNasCell(self._filters,self._ops,self._adj,name=self._name+"_first_cell")
        for i in range(self._blck_len-1):
            self._cell_list.append(NanoNasCell(self._filters,self._ops,self._adj,name=self._name+"_cell_"+str(i)))
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

class NanoNasModel(tf.Module):
    def __init__(self,ops,adj,name="nanonasmodel"):
        super(NanoNasModel,self).__init__(name=name)
        self._ops=ops
        self._adj=adj
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._convbn=ConvBN(128,(3,3),(1,1),name=self._name+"_convbn")
        self._cellblck_1=NanoNasBlock(filters=128,adj=self._adj,ops=self._ops,blck_len=3,name=self._name+"_blck_1")
        self._avgpool_1=tf.keras.layers.AveragePooling2D(name=self._name+"_avgpool_1")
        self._cellblck_2=NanoNasBlock(filters=256,adj=self._adj,ops=self._ops,blck_len=3,name=self._name+"_blck_2")
        self._avgpool_2=tf.keras.layers.AveragePooling2D(name=self._name+"_avgpool_2")
        self._cellblck_3=NanoNasBlock(filters=512,adj=self._adj,ops=self._ops,blck_len=3,name=self._name+"_blck_3")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._convbn(input_ts)
        x=self._cellblck_1(x)
        x=self._avgpool_1(x)
        x=self._cellblck_2(x)
        x=self._avgpool_2(x)
        x=self._cellblck_3(x)
        out_ts=self._gap(x)
        return out_ts