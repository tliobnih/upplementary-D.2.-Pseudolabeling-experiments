import os
import torchvision
from torchvision import transforms
from src.transforms import *

def get_digits_dataset(opt,file):
	if file == 'mnist':
		transform = transforms.Compose([Channel1to3_v0(),
									transforms.ToTensor(),
									transforms.Resize([32,32]),
									transforms.Normalize(
									mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
									add_channel_0()
         							])
		dataset = torchvision.datasets.MNIST(
				root = os.path.join(opt.data_root,file),
				train = True,
				transform = transform, 
				download = True
		)
	elif file == 'svhn':
		transform =  transforms.Compose([transforms.ToTensor(),
                                	# transforms.Resize([224,224]),
									transforms.Normalize(
									mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
                                	add_channel_1()])
		dataset = torchvision.datasets.SVHN(
				root = os.path.join(opt.data_root,file),
				split='train',
				transform = transform, 
				download = True
		)
	return dataset
