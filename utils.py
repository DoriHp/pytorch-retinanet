# -*- coding: utf-8 -*-
# @Author: devBao
# @Date:   2021-02-18 14:42:23
# @Last Modified by:   devBao
# @Last Modified time: 2021-02-18 16:13:00

import torch.nn as nn

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def load_checkpoint(model: nn.Module, weights: str, depth: int) -> nn.Module:
    """Loads already trained weights to initialized model.
    Args:
        model (nn.Module): Empty retinanet model
        weights (str): Path to checkpoint
        depth (int): ResNet depth
    Raises:
        KeyError: If current model and checkpoint layers are not matching.
    Returns:
        nn.Module : retinanet model.
    """         
    if weights.endswith(".pt"):  # pytorch format
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            ckpt = {
                k: v
                for k, v in ckpt.state_dict().items()
                if model.state_dict()[k].shape == v.shape
            } 
            model.load_state_dict(ckpt, strict=True)
            print("Resuming training from checkpoint in {}".format(weights))  
        except KeyError as e:
            s = (
                "%s is not compatible with depth %s. This may be due to model architecture differences or %s may be out of date. "
                "Please delete or update %s and try again, or use --weights '' to train from scratch."
                % (weights, str(depth), weights, weights)
            )
            raise KeyError(s) from e
        del ckpt
        return model
    else:
        return model