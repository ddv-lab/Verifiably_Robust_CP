import numpy as np
import os
import torch
from enum import Enum
from functools import cache
from vrcp_reg_config import *


class MPEDataset(torch.utils.data.Dataset):
    
    STATE_FNAME = "state"
    REWARD_FNAME = "rewards"
    FN_TRAJ = "_traj_{}.npy"
    FN_STATE_PREFIX = "state_prefix.npy"
    FN_REWARD_PREFIX = "reward_prefix.npy"
    FN_INIT_STATE = "state_initial.npy"

    def __init__(self, seeds, data_path, target_idx=None, synthetic=False, sample_n_traj=None, max_traj=None, device=CONFIG["global"]["device"], horizon=None, C=1000, traj=None, **kwargs):
        self.seeds = seeds
        self.data_path = data_path
        self.target_idx = target_idx
        self.device = device
        self.horizon = CONFIG[CFG_DATA]["suffix_len"]
        self.traj = CONFIG[CFG_DATA]["n_suffixes"]
        self.states = len(seeds)

    def _get_init_state(self, seed):
        f_state = str(os.path.join(self.data_path, "seed_{}", MPEDataset.FN_INIT_STATE)).format(seed)
        return f_state

    def _get_prefix(self, seed):
        f_state = str(os.path.join(self.data_path, "seed_{}", MPEDataset.FN_STATE_PREFIX)).format(seed)
        f_action = str(os.path.join(self.data_path, "seed_{}", MPEDataset.FN_REWARD_PREFIX)).format(seed)
        return (f_state, f_action)
        
    def _get_traj(self, seed, traj):
        # print(f"Getting seed: {seed} and trajectory: {traj}")
        f_state = str(os.path.join(self.data_path, "seed_{}", MPEDataset.STATE_FNAME + MPEDataset.FN_TRAJ)).format(seed, traj)
        f_action = str(os.path.join(self.data_path, "seed_{}", MPEDataset.REWARD_FNAME + MPEDataset.FN_TRAJ)).format(seed, traj)
        return (f_state, f_action)
    
    @cache
    def get_file(self, filepath):
        return torch.from_numpy(np.load(filepath))

    def get_prefix_items(self, f_prefix_state, f_prefix_action):
        return torch.from_numpy(np.load(f_prefix_state)), torch.tensor(np.load(f_prefix_action))
    
    def get_suffix_items(self, f_suffix_state, f_suffix_action):
        return torch.from_numpy(np.load(f_suffix_state)), torch.tensor(np.load(f_suffix_action))


class StateData(MPEDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return int(self.states * self.traj)
    
    def __getitem__(self, idx):
        state_idx, traj_idx = np.unravel_index(idx, (self.states, self.traj))
        f_initial_state = self._get_init_state(self.seeds[state_idx])
        initial_state = self.get_file(f_initial_state).numpy()
        return initial_state.astype(np.float32)
    

class RewardData(MPEDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return int(self.states * self.traj)
    
    def __getitem__(self, idx):
        state_idx, traj_idx = np.unravel_index(idx, (self.states, self.traj))
        _, f_rewards = self._get_traj(self.seeds[state_idx], traj_idx)
        rewards = self.get_file(f_rewards)
        total_reward = np.sum(rewards.numpy()[:, 0])
        return total_reward.astype(np.float32)


class CombinedData(MPEDataset):
    
    def __init__(self, *args, single_traj=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_data = StateData(*args, **kwargs)
        self.reward_data = RewardData(*args, **kwargs)
        self.single_traj = single_traj

    def __len__(self):
        if self.single_traj:
            return int(self.states)
        else:
            return int(self.states * self.traj)
    
    def __getitem__(self, idx):
        if self.single_traj:
            idx = int(idx * self.traj)
        return self.state_data[idx], self.reward_data[idx]


class Datasets(Enum):
    STATE = StateData
    REWARD = RewardData
    COMBINED = CombinedData
