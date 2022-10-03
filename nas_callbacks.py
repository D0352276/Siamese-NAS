import tensorflow as tf
import random
from nas_prcss import CellPth2Cell,CellPths2JSON,JSON2CellPths,SamplingCellPths,CellPthsTraining,SiameseRanking,CellPthsPredicting,RankingCellPths

class Stabilizer(tf.keras.callbacks.Callback):
    def __init__(self,whts_path="stabilizer.hdf5"):
        super(Stabilizer,self).__init__()
        self._whts_path=whts_path
        self._cur_loss=9999
    def on_epoch_end(self,epoch,logs={}):
        loss=logs.get('loss')
        if(loss<self._cur_loss):
            self._cur_loss=loss
            self.model.save_weights(self._whts_path)
        return
    def GetBestModel(self):
        self.model.load_weights(self._whts_path)
        return self.model

class WeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self,save_path):
        super(WeightsSaver,self).__init__()
        self._save_path=save_path
    def on_epoch_begin(self,epoch,logs={}):
        self.model.save_weights(self._save_path)
        return 

class BestSaver(tf.keras.callbacks.Callback):
    def __init__(self,data_dir,noec_predictor,ec_predictor,save_path,max_nodes,all_ops,serch_batch=1000):
        super(BestSaver,self).__init__()
        self._data_dir=data_dir
        self._noec_predictor=noec_predictor
        self._ec_predictor=ec_predictor
        self._save_path=save_path
        self._all_ops=all_ops
        self._max_nodes=max_nodes
        self._serch_batch=serch_batch
        self._cur_best_acc=0
    def on_epoch_end(self,epoch,logs={}):
        cell_pths=SamplingCellPths(self._data_dir,self._serch_batch,False)
        if(self._ec_predictor!=None):
            cell_pths=SiameseRanking(cell_pths,self._noec_predictor,self._ec_predictor,self._all_ops,self._max_nodes,k=30)
        else:
            CellPthsPredicting(cell_pths,self._noec_predictor,self._all_ops,self._max_nodes)
            cell_pths=RankingCellPths(cell_pths)
        cur_acc=CellPth2Cell(cell_pths[0])["gt_accuracy"]
        if(cur_acc>self._cur_best_acc):
            self._cur_best_acc=cur_acc
            self.model.save_weights(self._save_path)
        return 

class BatchTopSampler(tf.keras.callbacks.Callback):
    def __init__(self,data_dir,noec_predictor,ec_predictor,max_nodes,all_ops,budget=10,update_split=5,total_epochs=100,serch_batch=1000):
        super(BatchTopSampler,self).__init__()
        self._data_dir=data_dir
        self._noec_predictor=noec_predictor
        self._ec_predictor=ec_predictor
        self._act_js_path="act_pths.json"
        self._max_nodes=max_nodes
        self._all_ops=all_ops
        self._budget=budget
        self._update_split=update_split
        self._total_epochs=total_epochs
        self._serch_batch=serch_batch
        self._epoch_count=0
        self._cur_stage=0
        self._samples_per_update=[self._budget//self._update_split for i in range(self._update_split-1)]+\
                                 [self._budget-int(self._budget//self._update_split*(self._update_split-1))]
        self._update_epochs=[total_epochs//(self._update_split+1)*(i+1) for i in range(self._update_split)]
    def _UpdateChecking(self):
        self._epoch_count+=1
        try:
            idx=self._update_epochs.index(self._epoch_count)
            samples_count=self._samples_per_update[idx]
            self._cur_stage=idx+1
            return samples_count
        except:
            return 0
    def _MergeKCellPths(self,act_pths,cell_pths,k):
        target_len=len(act_pths)+k
        while(k>0):
            k_cell_pths=cell_pths[:k]
            act_pths=list(set(act_pths+k_cell_pths))
            del cell_pths[:k]
            k=target_len-len(act_pths)
        return act_pths
    def _UpdateActCellPths(self,samples_count):
        if(samples_count==0):return
        act_pths=JSON2CellPths(self._act_js_path)
        cell_pths=SamplingCellPths(self._data_dir,self._serch_batch)
        if(self._ec_predictor!=None):
            cell_pths=SiameseRanking(cell_pths,self._noec_predictor,self._ec_predictor,self._all_ops,self._max_nodes,k=30)
        else:
            CellPthsPredicting(cell_pths,self._noec_predictor,self._all_ops,self._max_nodes)
            cell_pths=RankingCellPths(cell_pths)
        act_pths=self._MergeKCellPths(act_pths,cell_pths,samples_count)
        act_pths=CellPthsTraining(act_pths)
        CellPths2JSON(act_pths,self._act_js_path)
        return
    def on_epoch_end(self,epoch,logs={}):
        samples_count=self._UpdateChecking()
        self._UpdateActCellPths(samples_count)
        return