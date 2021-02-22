import argparse
import torch
from torchvision import transforms

from utils import *

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
	parser.add_argument('--model_path', help='Path to model', type=str)
	parser.add_argument('--images_path',help='Path to images directory',type=str)
	parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
	parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
	parser = parser.parse_args(args)

	#dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
	dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
	# Create the model
	#retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
	config = dict({"scales": None, "ratios": None})
	config = load_config("config2.yaml", config)
	retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=False, ratios=config["ratios"], scales=config["scales"])

	retinanet, _, _ = load_ckpt(parser.model_path, retinanet)

	use_gpu = True

	if use_gpu:
		print("Using GPU for validation process")
		if torch.cuda.is_available():
			retinanet = torch.nn.DataParallel(retinanet.cuda())
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.training = False
	retinanet.eval()
	retinanet.module.freeze_bn()

	print(csv_eval.evaluate(dataset_val, retinanet, score_threshold=0.4, iou_threshold=float(parser.iou_threshold)))

if __name__ == '__main__':
	main()
