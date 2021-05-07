#

import torch

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	"""
	Compute circular correlation of two tensors.
	Parameters
	----------
	a: Tensor,
	b: Tensor

	Notes
	-----
	Input a and b should have the same dimensions.

	Returns
	-------

	"""
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))