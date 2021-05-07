import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dataset.datasets import *

class SimpleLossCompute(nn.Module):
    eps=1e-8
    def __init__(self, criterion, grad_accum, opt=None):
        """Loss function and optimizer for single device

        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        grad_accum: int
            number of batches to accumulate gradients
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(SimpleLossCompute, self).__init__()
        self.criterion = criterion
        self.opt = opt
        self.acc_loss = 0
        self.n_correct = 0
        self.norm_term = 0
        self.loss = 0
        self.batch_count = 0
        self.grad_accum = grad_accum

    def __enter__(self):
        self.batch_count = 0

    def __exit__(self, type, value, traceback):
        # if not enough batches accumulated and there are gradients not applied,
        # do one more step
        if self.batch_count > 0:
            self.step()

    @property
    def avg_loss(self):
        return (self.acc_loss + self.eps) / (self.norm_term + self.eps)

    @property
    def bits_per_pixel(self):
        return self.avg_loss() / np.log(2.)

    @property
    def accuracy(self):
        return (self.n_correct + self.eps) / (self.norm_term + self.eps)

    def reset_meter(self):
        self.n_correct = 0
        self.acc_loss = 0
        self.norm_term = 0

    def step(self):
        self.opt.step()
        self.opt.optimizer.zero_grad()

    def backward_and_step(self):
        self.loss.backward()
        self.batch_count += 1
        # accumulate self.grad_accum times then synchronize and update
        if self.batch_count == self.grad_accum:
            self.step()
            self.batch_count = 0

    def __call__(self, y_pred, y, norm):
        y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y.contiguous().view(-1)
        self.loss = self.criterion(y_pred, y)
        if self.opt is not None:
            self.backward_and_step()
        self.n_correct += (y_pred.max(dim=-1)[1] == y).sum().item()
        self.acc_loss += self.loss.item() * norm
        self.norm_term += norm
        # When bp, we use mean, when report loss, we multiply by norm
        return self.loss.item()

class MultiGPULossCompute(SimpleLossCompute):
    def __init__(self, criterion, ndev, grad_accum, model, opt=None):
        """Loss function and optimizer for multiple devices

        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        ndev: int
            number of devices used
        grad_accum: int
            number of batches to accumulate gradients
        model: torch.nn.Module
            model to optimizer (needed to iterate and synchronize all parameters)
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(MultiGPULossCompute, self).__init__(criterion, grad_accum, opt=opt)
        self.ndev = ndev
        self.model = model

    def step(self):
        # multi-gpu synchronize gradients
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.ndev
        self.opt.step()
        self.opt.optimizer.zero_grad()
