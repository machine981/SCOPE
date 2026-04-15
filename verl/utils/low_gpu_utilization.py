import torch
import time
import os
import sys
import time
import argparse


def gpu_info(gpu_id=0):
   gpu_status = os.popen('nvidia-smi -i {} | grep %'.format(gpu_id)).read().split('|')
   # | N/A   34C    P0    72W / 400W |  69193MiB / 81251MiB |      0%      Default |
   gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
   gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
   gpu_utilization = int(gpu_status[3].split('%')[0].strip())
   return gpu_power, gpu_memory, gpu_utilization


if __name__ == '__main__':
#    time.sleep(2400)
   
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "--gpu_number",
       required=False,
       default="1",
       type=int,
   )
   args = parser.parse_args()

   inputs1 = []
   inputs2 = []
   for i in range(args.gpu_number): #['0', '1', '2', '3', '4', '5', '6', '7']:
       device = torch.device('cuda:'+str(i))
       inputs1.append(torch.rand(1000, 1000, dtype=torch.float).to(device))
       inputs2.append(torch.rand(1000, 1000, dtype=torch.float).to(device))
       #inputs2.append(torch.ones(5, 10).to(device))

   check_time = time.time()
   not_busy_devices = range(args.gpu_number)
   while True:
       now_time = time.time()
       if (now_time - check_time) > 5:
           not_busy_devices = []
           for i in range(args.gpu_number):
               gpu_power, gpu_memory, gpu_utilization = gpu_info(i)
               if gpu_utilization < 70:
                   not_busy_devices.append(i)
           check_time = time.time()
       for i in not_busy_devices:
        #    c = inputs1[i] + inputs2[i]
        #    d = inputs1[i] * inputs1[i]
           f = inputs1[i] @ inputs2[i]