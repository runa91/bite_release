
import numpy as np
import random
import copy
import time
import warnings

from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes
# from configs.dog_breeds.dog_breed_class import get_partial_summary



class TwoDatasetSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, batch_size_half, size0, size1, shuffle=True, drop_last=True):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size_half, _int_classes) or isinstance(batch_size_half, bool) or \
                batch_size_half <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size_half*2))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        assert size0 >= size1
        self.batch_size_half = batch_size_half
        self.size0 = size0
        self.size1 = size1
        self.shuffle = shuffle
        self.n_batches = self.size1//batch_size_half
        self.drop_last = drop_last

    def get_description(self):
        description = "\
            This sampler samples equally from two different datasets"
        return description


    def __iter__(self):

        dataset0 = np.arange(self.size0)
        dataset1_init = np.arange(self.size1) + self.size0
        if self.shuffle:
            np.random.shuffle(dataset0)

        dataset1 = []
        for ind in range(self.size0 // self.size1 + 1):
            dataset1_part = dataset1_init.copy()
            if self.shuffle:
                np.random.shuffle(dataset1_part)
            dataset1.extend(dataset1_part)
        dataset0 = dataset0[0:self.n_batches*self.batch_size_half]
        dataset1 = dataset1[0:self.n_batches*self.batch_size_half]

        for ind_batch in range(self.n_batches):
            d0 = dataset0[ind_batch*self.batch_size_half:(ind_batch+1)*self.batch_size_half]
            d1 = dataset1[ind_batch*self.batch_size_half:(ind_batch+1)*self.batch_size_half]

            batch = list(d0) + list(d1)
            yield batch



    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        '''if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore'''
        return self.n_batches








