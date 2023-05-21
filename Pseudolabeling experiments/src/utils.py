import os
import glob
import torch

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_topk_ckpt(model,weight_path,save_name,topk=5):
    torch.save(model,os.path.join(weight_path,save_name))
    weight_list = sorted(
        glob.glob(os.path.join(weight_path,'*.pt')),key=lambda x: float(x[-10:-3]),reverse=True)

    if len(weight_list) > topk:
        os.remove(weight_list[-1])
             
def get_topk_ckpt(weight_path):
    checkpointfile = os.listdir(weight_path)
    folder_names = [float(name[-10:-3]) for name in checkpointfile] 
    c_idx = checkpointfile[folder_names.index(max(folder_names))]  
    return c_idx