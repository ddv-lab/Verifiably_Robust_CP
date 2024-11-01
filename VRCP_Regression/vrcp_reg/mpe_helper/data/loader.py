import numpy as np
from torch.utils.data import DataLoader
from mpe_helper.util.helper import set_seed, reset_seed
from vrcp_reg_config import *

class Loader:

    def __init__(self, dataset, data_path=None, seeds=None, batch_size=32, split=None, shuffle=None, rng_seed=0, **kwargs):
        if rng_seed:
            set_seed(rng_seed)
        else:
            reset_seed()
        self.kwargs = kwargs
        if data_path is None: 
            self.data_path = CONFIG[CFG_PATH]['data']
            self.kwargs['data_path'] = self.data_path
        self.seeds = seeds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_fn = dataset.value

    def _instantiate_loader(self):
        pass

    def get_datasets(self):
        pass

    def get_loaders(self):
        pass


class SplitLoader(Loader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = [ int(x) for x in (np.array(self.split) * len(self.seeds)) ]
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self._instantiate_loader()

    def _instantiate_loader(self):
        # Compute indices to split along
        idx = np.arange(len(self.seeds))
        train_idx, val_idx, test_idx = np.split(idx, self.split)
        # Get dataset objects
        self.train_data = self.dataset_fn(self.seeds[train_idx], **self.kwargs)
        self.val_data = self.dataset_fn(self.seeds[val_idx], **self.kwargs)
        self.test_data = self.dataset_fn(self.seeds[test_idx], **self.kwargs)
        # Create data loaders for each set
        NUM_WORKERS = 8
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle[0], num_workers=NUM_WORKERS)
        self.val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=self.shuffle[1], num_workers=NUM_WORKERS)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle[2], num_workers=NUM_WORKERS)

    def get_datasets(self):
        return self.train_data, self.val_data, self.test_data

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


class SingleLoader(Loader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.loader = None
        self._instantiate_loader()

    def _instantiate_loader(self):
        self.dataset = self.dataset_fn(self.seeds, **self.kwargs)
        NUM_WORKERS = 8
        self.loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle[0], num_workers=NUM_WORKERS)

    def get_datasets(self):
        return self.dataset

    def get_loaders(self):
        return self.loader