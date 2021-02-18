import os
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	Normalizer
from retinanet.augmentation import get_aug_map
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

# Tensorboard (experiment)
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 

from utils import *

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
	parser.add_argument('--batch-size', help='Number of input images per step', type=int, default=1)
	parser.add_argument('--num-workers', help='Number of worker used in dataloader', type=int, default=1)

	parser.add_argument('--multi-gpus', help='Allow to use multi gpus for training task', action='store_true')
	parser.add_argument('--snapshots', help='Location to save training snapshots', type=str, default="snapshots")

	parser.add_argument('--log-dir', help='Location to save training logs', type=str, default="logs")
	parser.add_argument('--expr-augs', help='Allow to use use experiment augmentation methods', action='store_true')
	parser.add_argument('--aug-methods', help='(Experiment) Augmentation methods to use, separate by comma symbol', type=str, default="rotate,hflip,brightness,contrast")
	parser.add_argument('--aug-prob', help='Probability of applying (experiment) augmentation in range [0.,1.]', type=float, default=0.5)

	parser = parser.parse_args(args)

	train_transforms = [Normalizer()]

	# Define transform methods
	if parser.expr_augs:
		aug_map = get_aug_map(p=parser.aug_prob)
		aug_methods = parser.aug_methods.split(",")
		for aug in aug_methods:
			if aug in aug_map.keys():
				train_transforms.append(aug_map[aug])
			else:
				print(f"{aug} is not available.")
	else:
		train_transforms.append(Augmenter())

	train_transforms.append(Resizer())

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
									transform=transforms.Compose(train_transforms))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
								  transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on COCO,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')

		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
								   transform=transforms.Compose(train_transforms))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
									 transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	use_gpu = True

	if use_gpu:
		print("Using GPU for training process")
		if torch.cuda.is_available():
			if parser.multi_gpus:
				print("Using multi-gpus for training process")
				retinanet = torch.nn.DataParallel(retinanet.cuda(), device_ids=[0,1])
			else:
				retinanet = torch.nn.DataParallel(retinanet.cuda())
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	# Tensorboard writer
	writer = SummaryWriter("logs")

	# Save snapshots dir
	if not os.path.exists(parser.snapshots):
		os.makedirs(parser.snapshots)

	for epoch_num in range(parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()

		epoch_loss = []
		epoch_csf_loss = []
		epoch_reg_loss = []

		for iter_num, data in enumerate(dataloader_train):
			try:
				optimizer.zero_grad()

				if torch.cuda.is_available():
					with torch.cuda.device(0):
						classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
				else:
					classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
					
				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss
				epoch_csf_loss.append(float(classification_loss))
				epoch_reg_loss.append(float(regression_loss))

				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				print(
					'\rEpoch: {}/{} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
						(epoch_num + 1), parser.epochs, (iter_num + 1), len(dataloader_train), float(classification_loss), float(regression_loss), np.mean(loss_hist)), end='')

				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		# writer.add_scalar("Loss/train", loss, epoch_num)

		if parser.dataset == 'coco':

			print('Evaluating dataset')

			coco_eval.evaluate_coco(dataset_val, retinanet)

		elif parser.dataset == 'csv' and parser.csv_val is not None:

			print('\nEvaluating dataset')

			APs = csv_eval.evaluate(dataset_val, retinanet)
			mAP = mean(APs[ap][0] for ap in APs.keys())
			print("mAP: %.3f" %mAP)
			writer.add_scalar("validate/mAP", mAP, epoch_num)

		_epoch_loss = np.mean(epoch_loss)
		_epoch_csf_loss = np.mean(epoch_reg_loss)
		_epoch_reg_loss = np.mean(epoch_reg_loss)

		scheduler.step(_epoch_loss)

		lr = get_lr(optimizer)
		writer.add_scalar("train/classification-loss", _epoch_csf_loss, epoch_num)
		writer.add_scalar("train/regression-loss", _epoch_reg_loss, epoch_num)
		writer.add_scalar("train/loss", _epoch_loss, epoch_num)
		writer.add_scalar("train/learning-rate", lr, epoch_num)

		torch.save(retinanet.module, os.path.join(parser.snapshots, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))

		print('\n')

	retinanet.eval()

	torch.save(retinanet, 'model_final.pt')

	writer.flush()

if __name__ == '__main__':
	main()
