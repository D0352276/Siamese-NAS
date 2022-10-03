import os
from json_io import JSON2Dict,Dict2JSON
from nas_prcss import SamplingCellPths,CellPthsTraining,CellPthsEstCodeTraining,CellPthsNasEstCodeTraining,RankingCellPthsByFLOPs
import matplotlib.pyplot as plt
import numpy as np

class CellsAnalyst:
    def __init__(self,cells_dir):
        self._cells_dir=cells_dir
        self._cells=self._InitCells(cells_dir)
        self._cells_len=len(self._cells)
        self._trained_cells=self._TrainedCells(self._cells)
        self._trained_cells_len=len(self._trained_cells)
    def _InitCells(self,cells_dir):
        cells=[]
        cells_name=os.listdir(cells_dir)
        for cell_name in cells_name:
            cell_path=self._cells_dir+"/"+cell_name
            cell_dict=JSON2Dict(cell_path)
            cells.append(cell_dict)
        return cells
    def _TrainedCells(self,cells):
        trained_cells=[]
        for cell_dict in cells:
            if(cell_dict.get("gt_accuracy",-1)!=-1):
                trained_cells.append(cell_dict)
        return trained_cells
    def TrainedGtAccs(self):
        gt_accs=list(map(lambda x:x["gt_accuracy"],self._trained_cells))
        return gt_accs
    def TrainedEstCodes(self):
        est_codes=list(map(lambda x:x["est_code"],self._trained_cells))
        return est_codes
    def EstCodeAnalystForNas201(self,save_path):
        dist_function=(lambda x,y: np.sqrt(np.sum((x-y)**2,axis=-1)))

        plt.figure()
        plt.xlabel("GT Acc",fontsize=12)
        plt.ylabel("Est Code",fontsize=12)
        # plt.xlim(0.9,0.94)
        # plt.ylim(0.25,0.35)
        gt_accs=self.TrainedGtAccs()
        gt_accs=np.array(gt_accs)
        est_codes=self.TrainedEstCodes()
        est_codes=np.array(est_codes)
        est_codes=1-np.mean(est_codes,axis=-1)/np.max(est_codes)

        # g1_idxes_1=np.where((gt_accs>0.25)&(gt_accs<0.8))
        # g1_idxes_2=np.where((est_codes>0.)&(est_codes<0.11))
        # g1_idxes_1=np.squeeze(g1_idxes_1,axis=0)
        # g1_idxes_2=np.squeeze(g1_idxes_2,axis=0)
        # g1_idxes=np.intersect1d(g1_idxes_1,g1_idxes_2)

        # g2_idxes_1=np.where((gt_accs>0.45)&(gt_accs<0.8))
        # g2_idxes_2=np.where((est_codes>0.11)&(est_codes<0.18))
        # g2_idxes=np.intersect1d(g2_idxes_1,g2_idxes_2)

        # g3_idxes=np.where((gt_accs>0)&(gt_accs<1))
        # g3_idxes=np.setdiff1d(g3_idxes,g1_idxes,assume_unique=False)
        # g3_idxes=np.setdiff1d(g3_idxes,g2_idxes,assume_unique=False)


        # g1_gt_accs=gt_accs[g1_idxes]
        # g1_est_codes=est_codes[g1_idxes]
        # g2_gt_accs=gt_accs[g2_idxes]
        # g2_est_codes=est_codes[g2_idxes]
        # g3_gt_accs=gt_accs[g3_idxes]
        # g3_est_codes=est_codes[g3_idxes]


        # plt.plot(g1_gt_accs,g1_est_codes,marker=".",linewidth=0,markersize=8,color=[254/255,67/255,101/255],alpha=0.2)
        # plt.plot(g2_gt_accs,g2_est_codes,marker=".",linewidth=0,markersize=8,color=[108/255,152/255,198/255],alpha=0.2)
        # plt.plot(g3_gt_accs,g3_est_codes,marker=".",linewidth=0,markersize=8,color="gray",alpha=0.2)

        
        gt_accs=np.expand_dims(gt_accs,axis=1)
        est_codes=np.expand_dims(est_codes,axis=1)
        data=np.concatenate([gt_accs,est_codes],axis=-1)
        clusters_cent,clusters=Kmeans(data,k=4,dist_function=dist_function)
        colors=[[254/255,67/255,101/255],[108/255,152/255,198/255],(227/255,23/255,13/255),(51/255,161/255,201/255)]
        for i,cluster in enumerate(clusters):
            color=colors[i]
            x=cluster[...,0]
            y=cluster[...,1]
            plt.plot(x,y,marker=".",linewidth=0,markersize=8,color=color,alpha=0.2)

        plt.savefig(save_path)
        return
    def EstCodeAndGtAccCorrelation(self,save_path):
        plt.figure()
        plt.xlabel("GT Acc",fontsize=12)
        plt.ylabel("Est Code",fontsize=12)
        # plt.xlim(0.9,0.94)
        # plt.ylim(0.25,0.35)
        
        # plt.xlim(0.85,0.9)
        # plt.ylim(0.15,0.35)

        gt_accs=self.TrainedGtAccs()
        gt_accs=np.array(gt_accs)
        est_codes=self.TrainedEstCodes()
        est_codes=np.array(est_codes)
        est_codes=1-np.mean(est_codes,axis=-1)/np.max(est_codes)

        g1_idxes=np.where((gt_accs>0.9)&(gt_accs<0.95))
        g2_idxes=np.where((gt_accs>0.7)&(gt_accs<0.9))
        g3_idxes=np.where((gt_accs<0.7))
        g1_gt_accs=gt_accs[g1_idxes]
        g1_est_codes=est_codes[g1_idxes]
        g2_gt_accs=gt_accs[g2_idxes]
        g2_est_codes=est_codes[g2_idxes]
        g3_gt_accs=gt_accs[g3_idxes]
        g3_est_codes=est_codes[g3_idxes]

        plt.plot(g1_gt_accs,g1_est_codes,marker=".",linewidth=0,markersize=8,color=[254/255,67/255,101/255],alpha=0.2)
        plt.plot(g2_gt_accs,g2_est_codes,marker=".",linewidth=0,markersize=8,color=[108/255,152/255,198/255],alpha=0.2)
        plt.plot(g3_gt_accs,g3_est_codes,marker=".",linewidth=0,markersize=8,color="gray",alpha=0.2)

        plt.savefig(save_path)
        return
    def TrainedEstCodeLen(self):
        _len=0
        for cell_dict in self._trained_cells:
            if(cell_dict.get("est_code",-1)==-1):
                continue
            _len+=1
        return _len
    def TrainedCellsLen(self):
        return self._trained_cells_len
    def TrainingCodes(self):
        training_codes=[]
        for cell_dict in self._trained_cells:
            training_code=cell_dict["est_code"]
            training_codes.append(training_code)
        return training_codes
    def GroundTruthAccuracy(self):
        gt_accs=[]
        for cell_dict in self._trained_cells:
            gt_accs.append(cell_dict["gt_accuracy"])
        return gt_accs
    def SvaeTrainedCells(self,save_dir):
        for i,cell_dict in enumerate(self._trained_cells):
            save_path=save_dir+"/"+str(cell_dict["id"])+".json"
            Dict2JSON(cell_dict,save_path)
        return 
    def SvaeCellsByFLOPs(self,save_dir,mflops_thres=50):
        for i,cell_dict in enumerate(self._cells):
            if(cell_dict["flops"]>mflops_thres):continue
            save_path=save_dir+"/"+str(cell_dict["id"])+".json"
            Dict2JSON(cell_dict,save_path)
        return 
    def TrainCells(self):
        all_cell_pths=SamplingCellPths(self._cells_dir)
        all_cell_pths=RankingCellPthsByFLOPs(all_cell_pths)
        CellPthsTraining(all_cell_pths)
        return 
    def TrainCellsEstCode(self):
        all_cell_pths=SamplingCellPths(self._cells_dir)
        all_cell_pths=RankingCellPthsByFLOPs(all_cell_pths)
        CellPthsEstCodeTraining(all_cell_pths)
        return
    def TrainNasEstCode(self):
        all_cell_pths=SamplingCellPths(self._cells_dir)
        all_cell_pths=RankingCellPthsByFLOPs(all_cell_pths)
        CellPthsNasEstCodeTraining(all_cell_pths)
        return
    
        
# cells_dir="data/tiny_nanobench_synflow"
# cells_analyst=CellsAnalyst(cells_dir)
# # print(cells_analyst.TrainedCellsLen())
# # cells_analyst.TrainNasEstCode()
# cells_analyst.EstCodeAndGtAccCorrelation("test.png")


# cells_dir="data/nasbench201_cifar100"
# cells_analyst=CellsAnalyst(cells_dir)
# cells_analyst.EstCodeAnalystForNas201("test.png")

# print(cells_analyst.TrainedCellsLen())
# cells_analyst.EstCodeAndGtAccCorrelation("analys.png")

cells_dir="data/nanobench"
cells_analyst=CellsAnalyst(cells_dir)
print(cells_analyst.TrainedCellsLen())
cells_analyst.TrainCells()