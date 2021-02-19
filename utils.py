# -*- coding: utf-8 -*-
# @Author: devBao
# @Date:   2021-02-18 14:42:23
# @Last Modified by:   devBao
# @Last Modified time: 2021-02-19 20:36:09

import os
import torch
import torch.nn as nn
import shutil
from collections import OrderedDict
import yaml

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def load_ckpt(checkpoint_fpath, model, optimizer):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(checkpoint_fpath)

	# Load state_dict that save with nn.DataParallel
	new_state_dict = OrderedDict()
	for k, v in checkpoint['state_dict'].items():
		name = k[7:]
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)	
	model.to(device)

	optimizer.load_state_dict(checkpoint['optimizer'])

	return model, optimizer, checkpoint['epoch']

def save_ckpt(state, is_best, checkpoint_dir, name):
	f_path = os.path.join(checkpoint_dir, name)
	torch.save(state, f_path)
	if is_best:
		best_fpath = os.path.join(checkpoint_dir, 'best_model.pt')
		# shutil.copyfile(f_path, best_fpath)
		# With the best weights, save additional weights-only file
		torch.save(state['state_dict'], best_fpath)

def load_config(config, result):
	try:
		with open('config.yaml') as f:
			result = dict()
			docs = yaml.load_all(f, Loader=yaml.FullLoader)
			for doc in docs:
				for k, v in doc.items():
					result[k] = v
	except Exception as e:
		print("An error occurred when loading config file. Error's detail: ", e)
	finally:
		return result