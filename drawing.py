import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import os
from json_io import JSON2Dict

class PltFig:
    def __init__(self,x_label,y_label,x_lim=None,y_lim=None):
        plt.figure()
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
        if(x_lim!=None):plt.xlim(*x_lim)
        if(y_lim!=None):plt.ylim(*y_lim)
        plt.grid()
    def _CurveSmooth(self,x_arr,y_arr):
        x_arr=x_arr.copy()
        y_arr=y_arr.copy()
        orig_len=len(x_arr)
        model=make_interp_spline(x_arr,y_arr)
        _x_arr=np.linspace(x_arr[0],x_arr[-1],int(orig_len*0.5))
        y_arr=model(_x_arr)
        model=make_interp_spline(_x_arr,y_arr)
        _x_arr=np.linspace(_x_arr[0],_x_arr[-1],orig_len)
        y_arr=model(_x_arr)
        return x_arr,y_arr
    def PlotCurve(self,x_arr,y_arr,alpha=1,linewidth=1,label="",color="gray",smooth=False):
        if(smooth==True):
            x_arr,y_arr=self._CurveSmooth(x_arr,y_arr)
        plt.plot(x_arr,y_arr,marker="",linewidth=linewidth,label=label,color=color,alpha=alpha)
        return
    def PlotCrtrnLine(self,x_val,y_lim,alpha=0.5,linewidth=2,color="gray"):
        plt.plot([x_val,x_val],y_lim,marker="",linestyle="--",linewidth=linewidth,color=color,alpha=alpha)
        return
    def PlotSymbol(self,x_arr,y_arr,marker="^",markersize=9,label="",color="gray"):
        plt.plot(x_arr,y_arr,marker=marker,markersize=markersize,linestyle="None",label=label,color=color)
        return
    def PlotStd(self,x_arr,y_arr,std_arr,std_scale=0.1,alpha=0.125,color="gray",smooth=False):
        if(smooth==True):
            x_arr,y_arr=self._CurveSmooth(x_arr,y_arr)
        std_arr=std_arr*std_scale
        plt.fill_between(x_arr,y_arr+std_arr,y_arr-std_arr,color=color,alpha=alpha)
        return
    def Save(self,save_path,legend_loc="upper left"):
        plt.legend(loc=legend_loc)
        plt.savefig(save_path)
        return
    
def GetTopN(results_dir,n=30,max_k=200):
    x_arr=[i for i in range(n+1,max_k+1,1)]
    x_arr=np.array(x_arr)
    y_arr=[]
    all_js=os.listdir(results_dir)
    for js_name in all_js:
        budgets=int(js_name.split("_")[0][1:])
        if(budgets!=n):continue
        js_path=results_dir+"/"+js_name
        js_dict=JSON2Dict(js_path)
        y_arr.append(js_dict["max_accs"][:max_k-n])
    y_arr=np.array(y_arr)
    y_mean=np.mean(y_arr,axis=0)
    y_std=np.std(y_arr,axis=0)
    return x_arr,y_mean,y_std

def GetTopK(results_dir,k=20):
    x_arr=[]
    all_js=os.listdir(results_dir)
    budget_dict={}
    for js_name in all_js:
        js_path=results_dir+"/"+js_name
        js_dict=JSON2Dict(js_path)
        budget=js_dict["budget"]
        if(budget not in budget_dict):
            budget_dict[budget]=[js_dict["max_accs"][k-1]]
        else:
            budget_dict[budget].append(js_dict["max_accs"][k-1])
    mean_vals=[]
    std_vals=[]
    budgets=list(budget_dict.keys())
    budgets.sort()
    x_arr=budgets
    for budget in budgets:
        vals=np.array(budget_dict[budget])
        mean_val=np.mean(vals)
        std_val=np.std(vals)
        std_vals.append(std_val)
        mean_vals.append(mean_val)
    mean_vals=np.array(mean_vals)
    std_vals=np.array(std_vals)
    y_mean=mean_vals
    y_std=std_vals
    x_arr=np.array(x_arr)
    return x_arr,y_mean,y_std

def DrawTopN(all_results_dir,n,x_lim,y_lim,save_path,legend_loc="upper left"):
    model_types=os.listdir(all_results_dir)
    model_types.sort()
    colors=[(128/255,128/255,105/255,0.5),(244/255,164/255,96/255),(227/255,23/255,13/255),(51/255,161/255,201/255)]
    pltfig=PltFig("trained samples ("+str(n)+"+K)","mean acc",x_lim=x_lim,y_lim=y_lim)
    smooth=True
    std_scale=0.25
    for i,model_type in enumerate(model_types):
        color=colors[i]
        x_arr,y_mean,y_std=GetTopN(all_results_dir+"/"+model_type,n=n)
        pltfig.PlotCurve(x_arr,y_mean,label=model_type,color=color,smooth=smooth)
        pltfig.PlotStd(x_arr,y_mean,y_std,std_scale,color=color,smooth=smooth)
    #------------------BRP-NAS------------------
    # pltfig.PlotCrtrnLine(125,y_lim)
    # pltfig.PlotSymbol([100,125,150,200],[0.737,0.738,0.738,0.7385],marker="^",markersize=9,label="brp-nas",color=[183/255,127/255,221/255])
    # -------------------------------------------
    pltfig.Save(save_path,legend_loc=legend_loc)
    return

def DrawTopK(all_results_dir,k,x_lim,y_lim,save_path,legend_loc="upper left"):
    model_types=os.listdir(all_results_dir)
    model_types.sort()
    colors=[(128/255,128/255,105/255,0.5),(244/255,164/255,96/255),(227/255,23/255,13/255),(51/255,161/255,201/255),[254/255,67/255,101/255],[108/255,152/255,198/255]]
    pltfig=PltFig("trained samples (N+"+str(k)+")","mean acc",x_lim=x_lim,y_lim=y_lim)
    smooth=False
    std_scale=0.25
    for i,model_type in enumerate(model_types):
        color=colors[i]
        x_arr,y_mean,y_std=GetTopK(all_results_dir+"/"+model_type,k=k)
        pltfig.PlotCurve(x_arr+k,y_mean,label=model_type,color=color,smooth=smooth)
        pltfig.PlotStd(x_arr+k,y_mean,y_std,std_scale,color=color,smooth=smooth)

    #------------------BRP-NAS------------------
    # pltfig.PlotCrtrnLine(125,y_lim)
    # pltfig.PlotSymbol([100,125,150,200],[0.737,0.738,0.738,0.7385],marker="^",markersize=9,label="brp-nas",color=[183/255,127/255,221/255])
    # -------------------------------------------
    pltfig.Save(save_path,legend_loc=legend_loc)
    return


if(__name__=="__main__"):
    all_results_dir="paper_results/nas201_100_results"
    DrawTopK(all_results_dir,k=10,x_lim=[20,210],y_lim=[0.71,0.75],save_path="paper_results/nas201_100_imgs/N_10.png")
    DrawTopK(all_results_dir,k=20,x_lim=[20,210],y_lim=[0.71,0.75],save_path="paper_results/nas201_100_imgs/N_20.png")
    DrawTopK(all_results_dir,k=30,x_lim=[20,210],y_lim=[0.71,0.75],save_path="paper_results/nas201_100_imgs/N_30.png")

    all_results_dir="paper_results/nas201_10_results"
    DrawTopK(all_results_dir,k=10,x_lim=[20,210],y_lim=[0.93,0.95],save_path="paper_results/nas201_10_imgs/N_10.png")
    DrawTopK(all_results_dir,k=20,x_lim=[20,210],y_lim=[0.93,0.95],save_path="paper_results/nas201_10_imgs/N_20.png")
    DrawTopK(all_results_dir,k=30,x_lim=[20,210],y_lim=[0.93,0.95],save_path="paper_results/nas201_10_imgs/N_30.png")

    all_results_dir="paper_results/lightweight_results"
    DrawTopK(all_results_dir,k=5,x_lim=[10,100],y_lim=[0.85,0.94],save_path="paper_results/lightweight_imgs/N_5.png")
    DrawTopK(all_results_dir,k=10,x_lim=[10,100],y_lim=[0.85,0.94],save_path="paper_results/lightweight_imgs/N_10.png")