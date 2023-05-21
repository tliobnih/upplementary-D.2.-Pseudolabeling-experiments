import os
import csv
import torch
from torchvision import transforms
from src.transforms import *
from src.utils import *
from tqdm import tqdm
import torch.nn.functional as nfunc

def inferences(seed,save_dir,target_file,train_data,infe_file=None,model=None):
    out_path = f'./dataset/pseudo_{target_file}_{seed}.csv'
    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    
    if model==None:
        checkpointfile_path = os.path.join(save_dir,infe_file)
        c_idx = get_topk_ckpt(checkpointfile_path)
        checkpoint = os.path.join(checkpointfile_path,c_idx)
        model = torch.load(checkpoint)
    model.to(device)
    
    result_file = open(os.path.join(out_path),mode='w',newline='')
    writer = csv.writer(result_file)
    writer.writerow(['image_name','label'])
    
    if target_file=='mnist':
        inference_transform = transforms.Compose([
                Channel1to3_v0(),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
    elif target_file=='svhn':
        inference_transform = transforms.Compose([
                change_channel(),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])

    inference_bar = tqdm(train_data.indices, desc=f'Make PseudoLabels')
    with torch.no_grad():
        model.eval()
        image_loc = train_data.dataset.data
        for i in inference_bar:
            image = image_loc[i]
            image = inference_transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            class_pred_x = model(image)
            class_pred = nfunc.softmax(class_pred_x, dim=1)
            class_pred_x.detach()
            class_pred = class_pred.argmax(dim=1).item()
            writer.writerow([i,class_pred])
        result_file.close()
