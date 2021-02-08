# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=7, verbose=False, delta=0):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.best_model = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.best_step = 0

	def __call__(self, val_loss, model, step = 0):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			# self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			# print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.best_model = copy.deepcopy(model.state_dict())
			# self.save_checkpoint(val_loss, model)
			self.counter = 0
			self.best_step = step

	# def save_checkpoint(self, val_loss, model):
	#     '''Saves model when validation loss decrease.'''
	#     if self.verbose:
	#         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
	#     torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
	#     self.val_loss_min = val_loss

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None,
				 patience = 20):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		self.patience = patience

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()
	def evaluate(self, data):
		with torch.no_grad():
			loss = self.model({
				'batch_h': self.to_var(data['batch_h'], self.use_gpu),
				'batch_t': self.to_var(data['batch_t'], self.use_gpu),
				'batch_r': self.to_var(data['batch_r'], self.use_gpu),
				'batch_y': self.to_var(data['batch_y'], self.use_gpu),
				'mode': data['mode']
			})
		return loss
	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		print('Early stopping. Patience',self.patience)
		earlyStopping = EarlyStopping(self.patience)
		# training_range = tqdm(range(self.train_times))
		for epoch in range(self.train_times):
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			
			val_loss = 0.0
			for data in self.data_loader:
				loss = self.evaluate(data)
				val_loss += loss 
			earlyStopping(val_loss,self.model,epoch)
			print("Epoch %d | loss: %f | val_loss: %f | best epoch: %d" % (epoch, res, val_loss,earlyStopping.best_step))
			
			if earlyStopping.early_stop:
				print('Early stopping. Best Epoch:',earlyStopping.best_step)
				self.model.load_state_dict(earlyStopping.best_model)
				self.model.save_checkpoint(self.checkpoint_dir)
				return
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.load_state_dict(earlyStopping.best_model)
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))
		print('Finish all epoch. Save Best Eporch:',earlyStopping.best_step)
		self.model.load_state_dict(earlyStopping.best_model)
		self.model.save_checkpoint(self.checkpoint_dir)

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir