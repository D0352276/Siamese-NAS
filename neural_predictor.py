import tensorflow as tf
from graph_conv import GConvGroup
from modules import NodesAttention,mish

class EmbedingBlock(tf.Module):
    def __init__(self,hid_len=64,nds_attn=False,name="embdblck"):
        super(EmbedingBlock,self).__init__(name=name)
        self._embd_len=hid_len
        self._nds_attn=nds_attn
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._gcnconv_1=GConvGroup(self._embd_len,use_bn=True,name=self._name+"_gcnconv_1")
        self._gcnconv_2=GConvGroup(self._embd_len,use_bn=True,name=self._name+"_gcnconv_2")
        self._ndsatten_1=NodesAttention(name=self._name+"_ndsatten_1")
        self._gcnconv_3=GConvGroup(self._embd_len,use_bn=True,name=self._name+"_gcnconv_3")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,x=input_ts
        x=self._gcnconv_1([adj,x])
        x=self._gcnconv_2([adj,x])
        if(self._nds_attn==True):
            x=self._ndsatten_1([adj,x])
        x=self._gcnconv_3([adj,x])
        out_ts=x
        return out_ts

class EstmtnFusion(tf.Module):
    def __init__(self,name="estmtnfusion"):
        super(EstmtnFusion,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,nodes_len,feats_len):
        self._ds_k=tf.keras.layers.Dense(nodes_len*feats_len,activation=None,name=self._name+"_ds_k")
        self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn1")
        self._gcnconv_q=GConvGroup(feats_len,use_bn=True,name=self._name+"_gcnconv_q")
        self._gcnconv_k=GConvGroup(feats_len,use_bn=True,name=self._name+"_gcnconv_k")
        self._gcnconv_v=GConvGroup(feats_len,use_bn=True,name=self._name+"_gcnconv_v")
        self._gconvout=GConvGroup(feats_len,name=self._name+"_gconvout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,feats,est_code=input_ts
        nodes_len,feats_len=feats.get_shape()[1:]
        self._Build(nodes_len,feats_len)
        k=self._ds_k(est_code)
        k=tf.reshape(k,[-1,nodes_len,feats_len])
        k=self._bn(k)
        k=mish(k)
        k=self._gcnconv_k([adj,k])
        q=self._gcnconv_q([adj,feats])
        v=self._gcnconv_q([adj,feats])
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

class SiamsPredictor(tf.Module):
    def __init__(self,nds_attn=False,name="siams_predictor"):
        super(SiamsPredictor,self).__init__(name=name)
        self._nds_attn=nds_attn
        self._name=self._name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._embd_blck=EmbedingBlock(nds_attn=self._nds_attn,name=self._name+"_embd_blck")
        self._estattn=EstmtnFusion(name=self._name+"_estattn")
        self._denseout=tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,name=self._name+"_denseout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,x,est_codes=input_ts
        x1=self._embd_blck([adj,x])
        x2=self._estattn([adj,x1,est_codes])
        x1=tf.reduce_mean(x1,axis=-1)
        x2=tf.reduce_mean(x2,axis=-1)
        x1_acc=self._denseout(x1)
        x2_acc=self._denseout(x2)
        return x1_acc,x2_acc

class NeuralPredictor(tf.Module):
    def __init__(self,hid_len=64,name="neural_predictor"):
        super(NeuralPredictor,self).__init__(name=name)
        self._hid_len=hid_len
        self._name=self._name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._gcnconv_1=GConvGroup(self._hid_len,use_bn=True,name=self._name+"_gcnconv_1")
        self._gcnconv_2=GConvGroup(self._hid_len,use_bn=True,name=self._name+"_gcnconv_2")
        self._gcnconv_3=GConvGroup(self._hid_len,use_bn=True,name=self._name+"_gcnconv_3")
        self._dsout_acc=tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,name=self._name+"_dsout_acc")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        adj,x=input_ts
        x=self._gcnconv_1([adj,x])
        x=self._gcnconv_2([adj,x])
        x=self._gcnconv_3([adj,x])
        x=tf.reduce_mean(x,axis=-1)
        acc=self._dsout_acc(x)
        return acc
    
class WeightedL1(tf.Module):
    def __init__(self, name="whted_l1"):
        super(WeightedL1, self).__init__(name)
        self._name=name
    def _AccsLoss(self,true_accs,pred_accs):
        huber_delta=0.5
        l1_loss=tf.math.abs(true_accs-pred_accs)
        l1_loss=tf.keras.backend.switch(l1_loss<huber_delta,0.5*l1_loss**2,huber_delta*(l1_loss-0.5*huber_delta))
        return l1_loss
    def __call__(self):
        def _WeightedL1(true_y,pred_y):
            acc_1=true_y[...,0:1]
            cnfd_1=true_y[...,1:2]
            acc_2=true_y[...,2:3]
            cnfd_2=true_y[...,3:4]
            lam=true_y[...,4:5]
            pred_acc=pred_y[...,0:1]
            loss_1=self._AccsLoss(acc_1,pred_acc)*cnfd_1*lam
            loss_2=self._AccsLoss(acc_2,pred_acc)*cnfd_2*(1-lam)
            loss=loss_1+loss_2
            return loss
        return _WeightedL1

def CreateNeuralPredictor(node_num,feat_dim,weights_path=None):
    feat_input_shape=(node_num,feat_dim)
    adj_input_shape=(node_num,node_num)
    feat_in_ts=tf.keras.Input(shape=feat_input_shape)
    adj_in_ts=tf.keras.Input(shape=adj_input_shape)
    ts_out=NeuralPredictor()([adj_in_ts,feat_in_ts])
    model=tf.keras.Model(inputs=[adj_in_ts,feat_in_ts],outputs=ts_out)
    if(weights_path!=None):
        model.load_weights(weights_path)
    return model

def CreateSiamesePredictor(node_num,feat_dim,nds_attn=False,weights_path=None):
    feat_input_shape=(node_num,feat_dim)
    adj_input_shape=(node_num,node_num)
    code_input_shape=(3)
    feat_in_ts=tf.keras.Input(shape=feat_input_shape)
    adj_in_ts=tf.keras.Input(shape=adj_input_shape)
    code_in_ts=tf.keras.Input(shape=code_input_shape)
    no_train_code_out,train_code_out=SiamsPredictor(nds_attn=nds_attn)([adj_in_ts,feat_in_ts,code_in_ts])
    model_with_code=tf.keras.Model(inputs=[adj_in_ts,feat_in_ts,code_in_ts],outputs=[train_code_out,no_train_code_out])
    model_without_code=tf.keras.Model(inputs=[adj_in_ts,feat_in_ts],outputs=no_train_code_out)
    if(weights_path!=None):
        try:model_with_code.load_weights(weights_path)
        except:model_without_code.load_weights(weights_path)
    return model_without_code,model_with_code

def CompilePredictor(model,lr=0.01):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=WeightedL1()())
    return model