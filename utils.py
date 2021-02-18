# -*- coding: utf-8 -*-
# @Author: devBao
# @Date:   2021-02-18 14:42:23
# @Last Modified by:   devBao
# @Last Modified time: 2021-02-18 15:07:35

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
