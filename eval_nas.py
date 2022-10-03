import random
from nas_prcss import SamplingCellPths,CellPths2Cells,RankingCellPths,CellPthsPredicting,RankingCellPthsByBRP,SiameseRanking

def SpearmanRank(cell_pths):
    pred_rank_cells=CellPths2Cells(cell_pths)
    pred_ranks=[]
    for i,cell in enumerate(pred_rank_cells):
        id=cell["id"]
        pred_ranks.append([id,i+1])

    cell_pths=RankingCellPths(cell_pths,"gt")

    gt_rank_cells=CellPths2Cells(cell_pths)
    gt_ranks=[]
    for i,cell in enumerate(gt_rank_cells):
        id=cell["id"]
        gt_ranks.append([id,i+1])
    pred_ranks=sorted(pred_ranks,key=lambda x:x[0])
    gt_ranks=sorted(gt_ranks,key=lambda x:x[0])

    ids_len=len(pred_ranks)
    ids_diff=0
    for i,elemt in enumerate(pred_ranks):
        _,pred_rank=elemt
        _,gt_rank=gt_ranks[i]
        id_diff=(pred_rank-gt_rank)**2
        ids_diff+=id_diff
    psp=1-((6*ids_diff)/(ids_len*(ids_len**2-1)))
    return psp

def MaxAccs(cell_pths):
    pred_rank_cells=CellPths2Cells(cell_pths)
    max_accs=[]
    cur_max_acc=0
    for cell in pred_rank_cells:
        gt_acc=cell["gt_accuracy"]
        if(gt_acc>cur_max_acc):
            cur_max_acc=gt_acc
        max_accs.append(cur_max_acc)
    return max_accs

def EvalPredictor(predictor,cells_dir,all_ops,max_nodes):
    all_cell_pths=SamplingCellPths(cells_dir)
    CellPthsPredicting(all_cell_pths,predictor,all_ops,max_nodes,False)
    all_cell_pths=RankingCellPths(all_cell_pths,"pred")
    max_accs=MaxAccs(all_cell_pths)
    psp=SpearmanRank(all_cell_pths)
    return max_accs,psp

def EvalRandom(cells_dir):
    all_cell_pths=SamplingCellPths(cells_dir)
    random.shuffle(all_cell_pths)
    max_accs=MaxAccs(all_cell_pths)
    psp=SpearmanRank(all_cell_pths)
    return max_accs,psp

def EvalSiamesePredictor(noec_predictor,ec_predictor,cells_dir,all_ops,max_nodes,k=10):
    all_cell_pths=SamplingCellPths(cells_dir)
    all_cell_pths=SiameseRanking(all_cell_pths,noec_predictor,ec_predictor,all_ops,max_nodes,k=k)
    psp=SpearmanRank(all_cell_pths)
    max_accs=MaxAccs(all_cell_pths)
    return max_accs,psp

def EvalBRPredictor(br_predictor,cells_dir,all_ops,max_nodes):
    all_cell_pths=SamplingCellPths(cells_dir,shuffle=False)
    all_cell_pths=RankingCellPthsByBRP(all_cell_pths,br_predictor,all_ops,max_nodes)
    max_accs=MaxAccs(all_cell_pths)
    psp=SpearmanRank(all_cell_pths)
    return max_accs,psp