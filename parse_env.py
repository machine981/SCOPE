#! coding: utf-8

import sys
import os
import json

def main(args):
    ''' 
        用于解析MLP中，worker的环境变量；此程序需要运行在HOPE的任务重
    '''
    env_name = args[1]
    #cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    cluster_spec = json.loads(os.environ["AFO_SPEC"])
    workers = cluster_spec["cluster"]["worker"]
    master = workers[0]
    master_addr, master_ports = master.split(":")
    master_port = master_ports.split(",")[0]
    nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
    nnodes = len(workers)
    node_rank = cluster_spec["taskId"]
    
    #print(cluster_spec)
    #print("nproc_per_node: {}".format(nproc_per_node))
    #print("master_addr: {}".format(master_addr))
    #print("master_port: {}".format(master_port))
    #print("nnodes: {}".format(nnodes))
    #print("node_rank: {}".format(node_rank))

    if env_name == 'nproc_per_node':
        return nproc_per_node
    elif env_name == 'master_addr':
        return master_addr
    elif env_name == 'master_port':
        return master_port
    elif env_name == 'nnodes':
        return nnodes
    elif env_name == 'node_rank':
        return node_rank

    raise "WRONG ARGUMENT"


if __name__ == '__main__':
    print(main(sys.argv))
