import tensorflow as tf
import cv2
import os
import random
from nas_prcss import SamplingCellPths,CellPth2Cell
from json_io import Dict2JSON
from nanonas_model import NanoNasModel
from nas201_model import Nas201Model

def SnipScore(model,optimizer,cifar_dir,k=8,label_len=10):
    loss_mean=0
    for i in range(10):
        imgs,labels=SamplingCIFAR10(cifar_dir,k)
        in_ts=tf.convert_to_tensor(imgs)
        label_ts=tf.Variable(labels)

        with tf.GradientTape() as gtape:
            pred=model(in_ts)
            one_hot_label=tf.one_hot(label_ts,label_len)
            ce_loss=tf.keras.losses.categorical_crossentropy(one_hot_label,tf.nn.softmax(pred))
            ce_loss=tf.reduce_sum(ce_loss)/k
            grads=gtape.gradient(ce_loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            loss_mean+=(ce_loss)
    score=loss_mean/10
    score=float(score.numpy())
    return score

def SynFlowScore(model,input_shape=[32,32,3],alpha=0.001):
    act_layers=0
    in_ts=tf.ones(input_shape)
    in_ts=tf.expand_dims(in_ts,axis=0)
    with tf.GradientTape() as gtape:
        pred=model(in_ts)
        pred=tf.reduce_sum(pred,axis=-1)
        grads=gtape.gradient(pred,model.trainable_variables)
        score=0
        for i,grad in enumerate(grads):
            if(type(grad)==type(None)):continue
            whts=model.trainable_variables[i]
            synflow=tf.abs(grad*whts)
            synflow=tf.reduce_sum(synflow)
            score+=synflow
            act_layers+=1
    score=score/act_layers
    score=tf.clip_by_value(score,0,int(1/alpha))*alpha
    synflow_score=float(score.numpy())
    return synflow_score

def SamplingCIFAR10(cifar_dir,k=32):
    labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    imgs_name=os.listdir(cifar_dir)
    chosen_imgs_name=random.choices(imgs_name,k=k)
    imgs=[]
    label_idxs=[]
    for chosen_img_name in chosen_imgs_name:
        label=chosen_img_name.split("_")[0]
        label_idx=labels.index(label)
        img_path=cifar_dir+"/"+chosen_img_name
        img=cv2.imread(img_path)/255
        imgs.append(img)
        label_idxs.append(label_idx)
    return imgs,label_idxs

def GetCellsSynFlowCode(cells_dir,cifar_dir,k=16,label_len=10,input_shape=[32,32,3]):
    in_ts=tf.keras.layers.Input(input_shape)
    cell_pths=SamplingCellPths(cells_dir)
    for idx,cell_pth in enumerate(cell_pths):
        cell=CellPth2Cell(cell_pth)
        # x=NanoNasModel(cell["operations"],cell["adj_matrix"])(in_ts)
        x=Nas201Model(cell["operations"],cell["adj_matrix"],cell["init_channel"],cell["blck_len"])(in_ts)
        out_ts=tf.keras.layers.Dense(label_len,activation=None)(x)
        model=tf.keras.Model(inputs=in_ts,outputs=out_ts)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        score=SynFlowScore(model)
        # score=SnipScore(model,optimizer,cifar_dir)
        cell["est_code"]=[score,score,score]
        Dict2JSON(cell,cell_pth)
    

