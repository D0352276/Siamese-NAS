import numpy as np
from json_io import Dict2JSON
from nats_bench import create

def NasBench201Adj():
    adj_m=np.zeros([10,10])
    connections=[[0,1],[0,3],[0,6],[1,2],[2,4],[2,7],[3,5],[4,5],[5,8],[6,9],[7,9],[8,9]]
    for connection in connections:
        start_idx,end_idx=connection
        adj_m[start_idx][end_idx]=1
    adj_m=adj_m.astype("int")
    return adj_m.tolist()

def ArchStr2Ops(arch_str):
    nodes_str=arch_str.split("+")
    nodes_str_list=[]
    for node_str in nodes_str:
        node_str_list=node_str.split("|")
        del node_str_list[0]
        del node_str_list[-1]
        node_str_list=list(map(lambda x:x.split("~")[0],node_str_list))
        node_str_list=list(map(lambda x:"zeros" if x=="none" else x,node_str_list))
        nodes_str_list=nodes_str_list+node_str_list
    ops=[]
    ops.append("split_3")
    ops.append(nodes_str_list[0])
    ops.append("split_2")
    ops.append(nodes_str_list[1])
    ops.append(nodes_str_list[2])
    ops.append("none")
    ops.append(nodes_str_list[3])
    ops.append(nodes_str_list[4])
    ops.append(nodes_str_list[5])
    ops.append("output")
    return ops

def NasBench201toJSONs(nasbench_file_path,out_dir,data_type="cifar10"):
    api=create(nasbench_file_path,'tss',fast_mode=True)
    adj_m=NasBench201Adj()
    for i in range(len(api)):
        if(i<10000):continue
        out_path=out_dir+"/"+str(i)+".json"
        info=api.get_more_info(i,data_type,hp="200")
        config=api.get_net_config(i,data_type)
        cell_dict={}
        ops=ArchStr2Ops(config["arch_str"])
        cell_dict["id"]=i
        cell_dict["adj_matrix"]=adj_m
        cell_dict["operations"]=ops
        cell_dict["init_channel"]=config["C"]
        cell_dict["blck_len"]=config["N"]
        cell_dict["gt_accuracy"]=info["test-accuracy"]/100
        cell_dict["pred_accuracy"]=-1
        cell_dict["flops"]=-1
        Dict2JSON(cell_dict,out_path)


# api=create('data/backup/NATS-tss-v1_0-3ffb9-simple','tss',fast_mode=True)
# info=api.get_more_info(4506,"cifar10",hp="200")
# info = api.get_cost_info(4506, 'cifar10')
# config=api.get_net_config(4506,"cifar10")
# print(info)