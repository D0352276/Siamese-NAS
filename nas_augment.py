import random
import numpy as np

def CellMixup(adj_mat_1,op_mat_1,adj_mat_2,op_mat_2):
    if(random.random()>0.5):
        lam=np.random.beta(1.5,1.5)
        adj_mat=lam*adj_mat_1+(1-lam)*adj_mat_2
        op_mat=lam*op_mat_1+(1-lam)*op_mat_2
    else:
        adj_mat=adj_mat_1
        op_mat=op_mat_1
        lam=1.0
    adj_mat=adj_mat_1
    op_mat=op_mat_1
    lam=1.0
    return adj_mat,op_mat,lam
