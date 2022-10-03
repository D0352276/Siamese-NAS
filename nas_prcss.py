import os
import numpy as np
import random
from json_io import Dict2JSON,JSON2Dict
from nas_training import TrainNanoNasModel,TrainEstCode,TrainNas201EstCode

def ADJMatrix(adj,max_nodes):
    pad_num=max_nodes-len(adj)
    for elemt in adj:
        for i in range(pad_num):elemt.append(0)
    for i in range(pad_num):adj.append([0 for j in range(max_nodes)])
    adj_matrix=np.array(adj)
    adj_matrix=adj_matrix+np.identity(max_nodes)
    return adj_matrix

def OPsMatrix(all_ops,chosen_ops,max_nodes):
    pad_num=max_nodes-len(chosen_ops)
    if(pad_num>0):
        for i in range(pad_num):chosen_ops.append('none')
    op_matrix=[]
    for op in chosen_ops:
        one_hot=[0 for i in range(len(all_ops))]
        one_hot[all_ops.index(op)]=1
        op_matrix.append(one_hot)
    return np.array(op_matrix)

def CellPths2JSON(cell_pths,save_path):
    js_dict={}
    js_dict["cell_pths"]=cell_pths
    Dict2JSON(js_dict,save_path)
    return

def JSON2CellPths(js_path):
    cell_pths=JSON2Dict(js_path)["cell_pths"]
    return cell_pths 

def CellPth2Cell(cell_pth,all_ops=[],max_nodes=7,preprcss=False):
    cell=JSON2Dict(cell_pth)
    if(preprcss==True):
        cell["adj_matrix"]=ADJMatrix(cell["adj_matrix"],max_nodes)
        cell["operations"]=OPsMatrix(all_ops,cell["operations"],max_nodes)
    return cell

def CellPths2Cells(cell_pths,all_ops=[],max_nodes=7,preprcss=False):
    return list(map(lambda x:CellPth2Cell(x,all_ops,max_nodes,preprcss),cell_pths))

def CellPthTraining(cell_pth):
    cell=CellPth2Cell(cell_pth)
    if(cell["gt_accuracy"]>0):return cell_pth
    gt_acc=TrainNanoNasModel(cell["operations"],cell["adj_matrix"])
    cell["gt_accuracy"]=gt_acc
    cell["confidence"]=1
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsTraining(cell_pths):
    return list(map(lambda x:CellPthTraining(x),cell_pths))

def CellPthEstCodeTraining(cell_pth):
    cell=CellPth2Cell(cell_pth)
    est_code=cell.get("est_code",None)
    if(type(est_code)==list):
        return cell_pth
    else:
        cell["est_code"]=TrainEstCode(cell["operations"],cell["adj_matrix"])
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsEstCodeTraining(cell_pths):
    return list(map(lambda x:CellPthEstCodeTraining(x),cell_pths))

def CellPthInit(cell_pth):
    cell=CellPth2Cell(cell_pth)
    cell["pred_accuracy"]=0
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsInit(cell_pths):
    return list(map(lambda x:CellPthInit(x),cell_pths))

def CellPthNasEstCodeTraining(cell_pth):
    cell=CellPth2Cell(cell_pth)
    est_code=cell.get("est_code",None)
    if(type(est_code)==list):
        return cell_pth
    else:
        cell["est_code"]=TrainNas201EstCode(cell["operations"],cell["adj_matrix"],cell["init_channel"],cell["blck_len"])
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsNasEstCodeTraining(cell_pths):
    return list(map(lambda x:CellPthNasEstCodeTraining(x),cell_pths))

def CellPthPredicting(cell_pth,predictor,all_ops=[],max_nodes=7,use_est_code=False):
    cell=CellPth2Cell(cell_pth,all_ops=all_ops,max_nodes=max_nodes,preprcss=True)
    if(use_est_code==True):
        preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]]),np.array([cell["est_code"]])))
    else:
        preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]])))
    pred_acc=preds[0][0]
    cell=CellPth2Cell(cell_pth,max_nodes=max_nodes,preprcss=False)
    cell["pred_accuracy"]=float(pred_acc)
    Dict2JSON(cell,cell_pth)
    return

def CellPthsPredicting(cell_pths,predictor,all_ops=[],max_nodes=7,use_est_code=False):
    for cell_pth in cell_pths:
        CellPthPredicting(cell_pth,predictor,all_ops,max_nodes,use_est_code)
    return

def RankingCellPths(cell_pths,rank_type="pred"):
    ranking_cell_pths=[]
    for cell_path in cell_pths:
        cell=CellPth2Cell(cell_path)
        gt_accuracy=cell["gt_accuracy"]
        pred_accuracy=cell["pred_accuracy"]
        if(rank_type=="gt"):
            CellPthTraining(cell_path)
            ranking_cell_pths.append([cell_path,gt_accuracy])
        else:
            ranking_cell_pths.append([cell_path,pred_accuracy])
    ranking_cell_pths=sorted(ranking_cell_pths,key=lambda x:x[1],reverse=True)
    ranking_cell_pths=list(map(lambda x:x[0],ranking_cell_pths))
    return ranking_cell_pths

def CellPthPredictingByBRP(cell_pth_1,cell_pth_2,br_predictor,all_ops=[],max_nodes=7):
    cell_1=CellPth2Cell(cell_pth_1,all_ops=all_ops,max_nodes=max_nodes,preprcss=True)
    cell_2=CellPth2Cell(cell_pth_2,all_ops=all_ops,max_nodes=max_nodes,preprcss=True)
    preds=br_predictor.predict_on_batch((np.array([cell_1["adj_matrix"]]),np.array([cell_1["operations"]]),np.array([cell_2["adj_matrix"]]),np.array([cell_2["operations"]])))
    pred_prob=preds[0]
    pred_idx=np.argmax(pred_prob)
    return pred_idx

def RankingCellPthsByBRP(cell_pths,br_predictor,all_ops=[],max_nodes=7):
    def _RankingCellPthsByBRP(cell_pth,ranking_cell_pths,begin_idx,end_idx,br_predictor,all_ops=[],max_nodes=7):
        mid_idx=begin_idx+(end_idx-begin_idx)//2
        if(mid_idx==begin_idx):
            pred_idx=CellPthPredictingByBRP(cell_pth,ranking_cell_pths[end_idx],br_predictor,all_ops,max_nodes)
            if(pred_idx==0):
                pred_idx=CellPthPredictingByBRP(cell_pth,ranking_cell_pths[begin_idx],br_predictor,all_ops,max_nodes)
                if(pred_idx==0):
                    ranking_cell_pths.insert(begin_idx,cell_pth)
                else:
                    ranking_cell_pths.insert(end_idx,cell_pth)
            else:
                ranking_cell_pths.append(cell_pth)
            return
        pred_idx=CellPthPredictingByBRP(cell_pth,ranking_cell_pths[mid_idx],br_predictor,all_ops,max_nodes)
        if(pred_idx==0):
            return _RankingCellPthsByBRP(cell_pth,ranking_cell_pths,begin_idx,mid_idx,br_predictor,all_ops,max_nodes)
        else:
            return _RankingCellPthsByBRP(cell_pth,ranking_cell_pths,mid_idx,end_idx,br_predictor,all_ops,max_nodes)
    pred_idx=CellPthPredictingByBRP(cell_pths[0],cell_pths[1],br_predictor,all_ops,max_nodes)
    if(pred_idx==0):ranking_cell_pths=[cell_pths[0],cell_pths[1]]
    if(pred_idx==1):ranking_cell_pths=[cell_pths[1],cell_pths[0]]

    cell_pths=cell_pths[2:]
    for cell_pth in cell_pths:
        _RankingCellPthsByBRP(cell_pth,ranking_cell_pths,0,len(ranking_cell_pths)-1,br_predictor,all_ops,max_nodes)
    return ranking_cell_pths

def RankingCellPthsByFLOPs(cell_pths):
    ranking_cell_pths=[]
    for cell_path in cell_pths:
        cell=CellPth2Cell(cell_path)
        flops=cell["flops"]
        ranking_cell_pths.append([cell_path,flops])
    ranking_cell_pths=sorted(ranking_cell_pths,key=lambda x:x[1],reverse=False)
    ranking_cell_pths=list(map(lambda x:x[0],ranking_cell_pths))
    return ranking_cell_pths

def SamplingCellPths(cells_dir,k=-1,shuffle=True):
    cells_list=[]
    all_cells=os.listdir(cells_dir)
    if(shuffle==True):random.shuffle(all_cells)
    if(k==-1):k=len(all_cells)
    act_count=0
    for cell_name in all_cells:
        cell_path=cells_dir+"/"+cell_name
        if(os.path.isfile(cell_path)!=True):continue
        cells_list.append(cell_path)
        act_count+=1
        if(act_count==k):break
    return cells_list

def SiameseRanking(cell_pths,noec_predictor,ec_predictor,all_ops,max_nodes,k=1):
    CellPthsPredicting(cell_pths,noec_predictor,all_ops,max_nodes,False)
    cell_pths=RankingCellPths(cell_pths,"pred")
    topk_cell_pths=cell_pths[:k]
    CellPthsPredicting(topk_cell_pths,ec_predictor,all_ops,max_nodes,True)
    topk_cell_pths=RankingCellPths(topk_cell_pths,"pred")
    cell_pths=topk_cell_pths+cell_pths[k:]
    return cell_pths