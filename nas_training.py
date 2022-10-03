import tensorflow as tf
from nanonas_model import NanoNasModel
from nas201_model import Nas201Model
from model_operation import Training
from eval_cifar10 import EvalCifar10
from json_io import JSON2Dict
import cv2
from cifar_augment import MixupAugment
import random
import numpy as np
import os

class NasModelDataGenerator:
    def __init__(self,data_dir,labels,target_hw=[32,32]):
        self._data_dir=data_dir
        self._img_names=os.listdir(data_dir)
        self._labels=labels
        self._labels_len=len(labels)
        self._target_hw=target_hw
        self._img_hw=None
    def _ReadData(self,img_name):
        img=cv2.imread(self._data_dir+"/"+img_name)/255
        label=img_name.split("_")[0]
        label_idx=self._labels.index(label)
        return img,label_idx
    def Read(self,batch_size=16):
        imgs=[]
        labels=[]
        chosen_img_names_1=random.choices(self._img_names,k=batch_size)
        chosen_img_names_2=random.choices(self._img_names,k=batch_size)
        for i,img_name_1 in enumerate(chosen_img_names_1):
            img_name_2=chosen_img_names_2[i]
            img_1,label_idx_1=self._ReadData(img_name_1)
            img_2,label_idx_2=self._ReadData(img_name_2)
            img,lam=MixupAugment(img_1,img_2,target_hw=self._target_hw)
            imgs.append(img)
            labels.append([label_idx_1,label_idx_2,lam])
        return np.array(imgs),np.array(labels)
    def Generator(self,batch_size=16):
        while(1):
            yield self.Read(batch_size)

class LossRecorder(tf.keras.callbacks.Callback):
    def __init__(self):
        self._records=[]
    def on_epoch_end(self,epoch,logs=None):
        self._records.append(logs["loss"])
    def GetRecords(self):
        return self._records

class MixupLoss(tf.Module):
    def __init__(self,name="mixuploss"):
        super(MixupLoss,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self):
        def _MixupLoss(true_y,pred_y):
            labels_len=tf.shape(pred_y)[-1]
            label_idx_1=tf.cast(true_y[...,0],tf.int32)
            label_idx_2=tf.cast(true_y[...,1],tf.int32)
            lam=tf.cast(true_y[...,2],tf.float32)
            one_hot_label_1=tf.one_hot(label_idx_1,labels_len)
            one_hot_label_2=tf.one_hot(label_idx_2,labels_len)
            bce_loss_1=tf.keras.losses.categorical_crossentropy(one_hot_label_1,pred_y,label_smoothing=0.01)*lam
            bce_loss_2=tf.keras.losses.categorical_crossentropy(one_hot_label_2,pred_y,label_smoothing=0.01)*(1-lam)
            loss=bce_loss_1+bce_loss_2
            return loss
        return _MixupLoss

def CompileModel(model,lr=0.001):
    adam=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam,
                loss=MixupLoss()())
    return model

def GetLossRecords(record_path,top_k=10):
    records=[]
    record_dict=JSON2Dict(record_path)
    records=record_dict["records"]
    return records[:top_k]

def TrainNanoNasModel(ops,adj):
    # hyper args
    labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    labels_len=len(labels)
    input_shape=[32,32,3]
    train_data_dir="data/cifar10/train"
    test_data_dir="data/cifar10/test"
    batch_size=128
    step_per_epoch=50000//batch_size

    in_ts=tf.keras.layers.Input(input_shape)
    out_ts=NanoNasModel(ops,adj)(in_ts)
    out_ts=tf.keras.layers.Dense(labels_len,activation=tf.nn.softmax)(out_ts)
    model=tf.keras.Model(inputs=in_ts,outputs=out_ts)
    model.summary()

    # training
    data_gen=NasModelDataGenerator(train_data_dir,labels,target_hw=input_shape[:2]).Generator(batch_size=batch_size)
    model=CompileModel(model,lr=0.01)
    Training(model,data_gen,step_per_epoch=step_per_epoch,epochs=40)
    model=CompileModel(model,lr=0.001)
    Training(model,data_gen,step_per_epoch=step_per_epoch,epochs=30)
    model=CompileModel(model,lr=0.0001)
    Training(model,data_gen,step_per_epoch=step_per_epoch,epochs=10)
    test_acc=EvalCifar10(model,test_data_dir)
    return test_acc

def TrainEstCode(ops,adj):
    # hyper args
    labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    labels_len=len(labels)
    input_shape=[32,32,3]
    train_data_dir="data/cifar10/tiny_train"
    batch_size=16
    step_per_epoch=len(os.listdir(train_data_dir))//batch_size
    
    data_gen=NasModelDataGenerator(train_data_dir,labels,target_hw=input_shape[:2])
    loss_recorder=LossRecorder()
    data_gen=data_gen.Generator(batch_size=batch_size)

    in_ts=tf.keras.layers.Input(input_shape)
    out_ts=NanoNasModel(ops,adj)(in_ts)
    out_ts=tf.keras.layers.Dense(labels_len,activation=tf.nn.softmax)(out_ts)
    model=tf.keras.Model(inputs=in_ts,outputs=out_ts)
    model=CompileModel(model,lr=0.001)
    Training(model,data_gen,step_per_epoch=step_per_epoch,epochs=3,callbacks=[loss_recorder])
    records=loss_recorder.GetRecords()
    return records

def TrainNas201EstCode(ops,adj,init_channel,blck_len):
    # hyper args
    labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    labels_len=len(labels)
    input_shape=[32,32,3]

    train_data_dir="data/cifar10/tiny_train"
    batch_size=16

    step_per_epoch=len(os.listdir(train_data_dir))//batch_size
    
    data_gen=NasModelDataGenerator(train_data_dir,labels,target_hw=input_shape[:2])
    loss_recorder=LossRecorder()
    data_gen=data_gen.Generator(batch_size=batch_size)

    in_ts=tf.keras.layers.Input(input_shape)
    out_ts=Nas201Model(ops,adj,init_channel=init_channel,blck_len=blck_len)(in_ts)
    out_ts=tf.keras.layers.Dense(labels_len,activation=tf.nn.softmax)(out_ts)
    model=tf.keras.Model(inputs=in_ts,outputs=out_ts)
    model=CompileModel(model,lr=0.001)
    Training(model,data_gen,step_per_epoch=step_per_epoch,epochs=3,callbacks=[loss_recorder])
    records=loss_recorder.GetRecords()
    return records