import numpy as np
import random
from nas_prcss import CellPths2JSON,JSON2CellPths,CellPth2Cell,SamplingCellPths,CellPthsTraining,CellPthEstCodeTraining,CellPthNasEstCodeTraining,CellPthsInit
from nas_augment import CellMixup

class NanoNasDataGenerator:
    def __init__(self,data_dir,all_ops,max_nodes=7,init_cells=10):
        self._data_dir=data_dir
        self._act_js_path="act_pths.json"
        self._all_ops=all_ops
        self._max_nodes=max_nodes
        self._InitActCellPths(init_cells)
    def _InitActCellPths(self,init_cells):
        cell_pths=SamplingCellPths(self._data_dir,init_cells)
        CellPthsInit(cell_pths)
        cell_pths=CellPthsTraining(cell_pths)
        CellPths2JSON(cell_pths,self._act_js_path)
        return
    def _CellDict(self,cell_path):
        CellPthEstCodeTraining(cell_path)
        cell_dict=CellPth2Cell(cell_path,self._all_ops,self._max_nodes,preprcss=True)
        return cell_dict
    def Read(self,batch_size=16,use_est_code=False):
        act_cell_pths=JSON2CellPths(self._act_js_path)
        act_cell_pths=random.choices(act_cell_pths,k=batch_size)
        act_cell_pths_2=random.choices(act_cell_pths,k=batch_size)
        adj_matrix_list=[]
        op_matrix_list=[]
        acc_cnfd_list=[]
        est_codes=[]
        for i,cell_path in enumerate(act_cell_pths):
            cell_path_2=act_cell_pths_2[i]
            cell_dict=self._CellDict(cell_path)
            cell_dict_2=self._CellDict(cell_path_2)
            gt_acc_1=cell_dict["gt_accuracy"]
            adj_mat_1=cell_dict["adj_matrix"]
            ops_mat_1=cell_dict["operations"]
            cnfd_1=cell_dict["confidence"]

            gt_acc_2=cell_dict_2["gt_accuracy"]
            adj_mat_2=cell_dict_2["adj_matrix"]
            ops_mat_2=cell_dict_2["operations"]
            cnfd_2=cell_dict_2["confidence"]

            adj_mat,op_mat,lam=CellMixup(adj_mat_1,ops_mat_1,adj_mat_2,ops_mat_2)
            adj_matrix_list.append(adj_mat)
            op_matrix_list.append(op_mat)
            acc_cnfd_list.append([gt_acc_1,cnfd_1,gt_acc_2,cnfd_2,lam])

            if(use_est_code==True):
                est_code_1=np.array(cell_dict["est_code"])
                est_code_2=np.array(cell_dict_2["est_code"])
                est_code=est_code_1*lam+est_code_2*(1-lam)
                est_codes.append(est_code)
                output_xy=(np.array(adj_matrix_list),np.array(op_matrix_list),np.array(est_codes)),np.array(acc_cnfd_list)
            else:
                output_xy=(np.array(adj_matrix_list),np.array(op_matrix_list)),np.array(acc_cnfd_list)

        return output_xy
    def Gen(self,batch_size=16,use_est_code=False):
        while(1):
            yield self.Read(batch_size,use_est_code)

class Nas101DataGenerator(NanoNasDataGenerator):
    def __init__(self,data_dir,all_ops,max_nodes=7,init_cells=10):
        super(Nas101DataGenerator,self).__init__(data_dir,all_ops,max_nodes,init_cells)

class Nas201DataGenerator(NanoNasDataGenerator):
    def __init__(self,data_dir,all_ops,max_nodes=7,init_cells=10):
        super(Nas201DataGenerator,self).__init__(data_dir,all_ops,max_nodes,init_cells)
    def _CellDict(self,cell_path):
        CellPthNasEstCodeTraining(cell_path)
        cell_dict=CellPth2Cell(cell_path,self._all_ops,self._max_nodes,preprcss=True)
        return cell_dict