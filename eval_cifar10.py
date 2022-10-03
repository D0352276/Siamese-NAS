import numpy as np
import random
import cv2
import os
from model_operation import CifarPredict

def EvalCifar10(model,test_data_dir,target_hw=[32,32]):
    labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    correct_count=0
    img_names=os.listdir(test_data_dir)
    random.shuffle(img_names)
    for i,img_name in enumerate(img_names):
        gt_label=img_name.split("_")[0]
        img=cv2.imread(test_data_dir+"/"+img_name)/255
        img=cv2.resize(img,(target_hw[1],target_hw[0]))
        pred_label=CifarPredict(model,labels,img)
        if(pred_label==gt_label):
            correct_count+=1
        print("Evaluating......"+str(i)+"th img, "+"current acc = "+str(correct_count/(i+1)))
    acc=correct_count/len(img_names)
    print("Total Accuracy = "+str(acc))
    return acc

def MAPE(pred_y,true_y):
    pred_y=np.array(pred_y)
    true_y=np.array(true_y)
    pred_y=np.reshape(pred_y,[-1])
    true_y=np.reshape(true_y,[-1])
    mape=np.sum(np.abs((pred_y-true_y)/true_y))/np.shape(true_y)[0]
    return mape

def CallbackEvalFunction(model,test_x,test_y):
    pred_y=model.predict(test_x)
    mape=MAPE(pred_y,test_y)
    print("\n\nMAPE: "+str(mape)+"\n")
    return 1-mape