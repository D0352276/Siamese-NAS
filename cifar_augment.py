import numpy as np
import cv2
import random

def FlipImg(img):
    img_w=np.shape(img)[1]
    img=cv2.flip(img,1)
    return img

def CropImg(img,crop_rate=0.1):
    h,w=np.shape(img)[:2]
    min_y=int(h*crop_rate)
    min_x=int(w*crop_rate)
    max_y=int(h*(1-crop_rate))
    max_x=int(w*(1-crop_rate))
    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y
    crop_xmin=max(0,int(min_x-random.uniform(0,max_l_trans)))
    crop_ymin=max(0,int(min_y-random.uniform(0,max_u_trans)))
    crop_xmax=max(w,int(max_x+random.uniform(0,max_r_trans)))
    crop_ymax=max(h,int(max_y+random.uniform(0,max_d_trans)))
    img=img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
    return img

def AffineImg(img,affine_rate=0.1):
    h,w=np.shape(img)[:2]
    min_y=int(h*affine_rate)
    min_x=int(w*affine_rate)
    max_y=int(h*(1-affine_rate))
    max_x=int(w*(1-affine_rate))

    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y

    tx=random.uniform(-(max_l_trans-1),(max_r_trans-1))
    ty=random.uniform(-(max_u_trans-1),(max_d_trans-1))

    M=np.array([[1,0,tx],[0,1,ty]])
    img=cv2.warpAffine(img,M,(w,h))
    return img

def ImgAugment(img):
    if(random.random()>0.5):
        img=FlipImg(img)
    if(random.random()>0.5):
        img=CropImg(img)
    if(random.random()>0.5):
        img=AffineImg(img)
    return img

def Resize(img,target_hw):
    img=cv2.resize(img,(target_hw[1],target_hw[0]))
    return img 

def Mixup(img1,img2):
    if(random.random()>0.5):
        lam=np.random.beta(1.5,1.5)
        img=lam*img1+(1-lam)*img2
    else:
        img=img1
        lam=1.0
    return img,lam
    
def MixupAugment(img1,img2,target_hw):
    img1=ImgAugment(img1)
    img2=ImgAugment(img2)
    img1=cv2.resize(img1,(target_hw[1],target_hw[0]))
    img2=cv2.resize(img2,(target_hw[1],target_hw[0]))
    img,lam=Mixup(img1,img2)
    return img,lam
