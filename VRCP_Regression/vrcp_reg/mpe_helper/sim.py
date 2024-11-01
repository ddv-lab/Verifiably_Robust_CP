from copy import deepcopy
from datetime import datetime
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import os
from typing import MutableSequence, Optional, Tuple

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from mpe_helper.environments import simple_spread, simple_adversary, simple_push
from vrcp_reg_config import *


class MPESim:

    def __init__(self, args, world_seed=None, noise_seed=None):
        self.name = "simple_spread"
        self.world_seed = CONFIG[CFG_RAND]['sim']['world_seed']
        self.noise_seed = CONFIG[CFG_RAND]['sim']['noise_seed']
        self.initial_state = None
        # Populated later
        self.state_shape = None
        self.action_shape = None
        # Create all the appropriate directories if they don't exist
        self.state_space_dimen = (CONFIG[CFG_ENV]['n_total_agents'] * 4) + (CONFIG[CFG_ENV]['n_landmarks'] * 2)
        self.max_cycles = CONFIG[CFG_DATA]['trajectory_len']
        self._setup_dirs()

        # Setup seeds
        self.world_rng = np.random.default_rng(seed=self.world_seed)
        self.noise_rng = np.random.default_rng(seed=self.noise_seed)

        # Get the policies, optimiser and the environment agents
        self.policy, self.optim, self.agents = self.get_agents()
        self.state_dimen = self.state_shape[0] if type(self.state_shape) is tuple else 1
        self.action_dimen = self.action_shape[0] if type(self.action_shape) is tuple else 1
        self.policy_snapshots = 0

    def _setup_dirs(self):
        paths = [CONFIG[CFG_PATH]['data'], CONFIG[CFG_PATH]['policy'], CONFIG[CFG_PATH]['model']]
        for path in paths:
            if not os.path.exists(path): os.makedirs(path)

    def get_env(
        self,
        initial_state: Optional[np.ndarray] = None,
    ) -> PettingZooEnv:
        # Returns customised simple adversary environment
        if CONFIG[CFG_PATH]['env_name'] == "spread":
            env_func = simple_spread.env
        elif CONFIG[CFG_PATH]['env_name'] == "adversary":
            env_func = simple_adversary.env
        elif CONFIG[CFG_PATH]['env_name'] == "push":
            env_func = simple_push.env
        else:
            raise ValueError("Env name must be either spread, adversary or push!")
        return PettingZooEnv(env_func(seed_w=self.world_seed, seed_n=self.noise_seed, max_cycles=self.max_cycles, continuous_actions=False, initial_state=initial_state))

    def get_agents(
        self,
    ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:

        def load_policies(
            agents: MutableSequence[BasePolicy]
        ) -> None:
            for agent, name in zip(agents, env.agents):
                policy_file = os.path.join(CONFIG[CFG_PATH]['policy'], f"{name}_final.pth")
                # If there exists a policy file and this isn't a random policy, then load the policy into agent's state
                if os.path.exists(policy_file) and not isinstance(agent, RandomPolicy):
                    agent.load_state_dict(torch.load(policy_file))
                    print(f"[ OK ] Loaded policy for {name}")
                else:
                    print(f"[WARN] Policy for {name} could not be found. Is this intentional?")

        env = self.get_env()
        observation_space = env.observation_space['observation'] if isinstance(
            env.observation_space, gym.spaces.Dict
        ) else env.observation_space
        self.state_shape = observation_space.shape or observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n

        # Create as many DQN policies as we have good agents
        agents = []
        print(f"NUM AGENTS: {Config.DATA[CFG_ENV]['n_total_agents']}  --  {env.agents}")
        for _ in range(CONFIG[CFG_ENV]['n_total_agents']):
            net = Net(
                self.state_shape,
                self.action_shape,
                hidden_sizes=CONFIG[CFG_SIM]['policy']['hidden_sizes'],
                device=CONFIG[CFG_GLOBAL]['device']
            ).to(CONFIG[CFG_GLOBAL]['device'])
            optim = torch.optim.Adam(net.parameters(), lr=CONFIG[CFG_SIM]['policy']['lr'])
            agents.append(DQNPolicy(net, optim, CONFIG[CFG_SIM]['policy']['gamma'], CONFIG[CFG_SIM]['policy']['n_step'], target_update_freq=CONFIG[CFG_SIM]['policy']['target_update_freq']))

        # Load policies for all agents if path provided
        if CONFIG[CFG_ENV]['load_policies']: load_policies(agents)
        
        policy = MultiAgentPolicyManager(agents, env)
        return policy, optim, env.agents

    def train_agents(
        self,
    ) -> Tuple[dict, MutableSequence[BasePolicy]]:

        # Setup environment
        train_envs = DummyVectorEnv([self.get_env for _ in range(CONFIG[CFG_SIM]['policy']['training_num'])])
        test_envs = DummyVectorEnv([self.get_env for _ in range(CONFIG[CFG_SIM]['policy']['test_num'])])
        # Seed the environments with seed specified in args
        # np.random.seed(CONFIG[CFG_RAND]['default_seed'])
        # torch.manual_seed(CONFIG[CFG_RAND]['default_seed'])
        # train_envs.seed(CONFIG[CFG_RAND]['default_seed'])
        # test_envs.seed(CONFIG[CFG_RAND]['default_seed'])

        # Setup data collectors
        train_collector = Collector(
            self.policy,
            train_envs,
            VectorReplayBuffer(CONFIG[CFG_SIM]['buffer_size'], len(train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(self.policy, test_envs, exploration_noise=True)
        train_collector.collect(n_step=CONFIG[CFG_SIM]['policy']['batch_size'] * CONFIG[CFG_SIM]['policy']['training_num'])

        # Setup tensorboard logging
        # Here we define the output folder.
        writer = SummaryWriter(CONFIG[CFG_PATH]['log'])
        logger = TensorboardLogger(writer)

        # Callback functions used during training
        def save_best_fn(policy, suffix=None):
            self.policy_snapshots += 1
            if not suffix: suffix = self.policy_snapshots
            for agent_idx in range(len(self.agents)):
                model_save_path = os.path.join(CONFIG[CFG_PATH]['policy'], f"{self.agents[agent_idx]}_{suffix}.pth")
                torch.save(policy.policies[self.agents[agent_idx]].state_dict(), model_save_path)

        # TODO: Think about what this means in the context of the simple adversary environment
        def stop_fn(mean_rewards):
            return False

        def train_fn(epoch, env_step):
            for agent_idx in range(CONFIG[CFG_ENV]['n_total_agents']):
                self.policy.policies[self.agents[agent_idx]].set_eps(CONFIG[CFG_SIM]['policy']['eps_train'])

        def test_fn(epoch, env_step):
            for agent_idx in range(CONFIG[CFG_ENV]['n_total_agents']):
                self.policy.policies[self.agents[agent_idx]].set_eps(CONFIG[CFG_SIM]['policy']['eps_test'])

        def reward_metric(rews):
            return rews[:, :CONFIG[CFG_ENV]['n_total_agents']]

        # Trainer
        result = offpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            CONFIG[CFG_SIM]['policy']['epoch'],
            CONFIG[CFG_SIM]['policy']['step_per_epoch'],
            CONFIG[CFG_SIM]['policy']['step_per_collect'],
            CONFIG[CFG_SIM]['policy']['test_num'],
            CONFIG[CFG_SIM]['policy']['batch_size'],
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=CONFIG[CFG_SIM]['policy']['update_per_step'],
            logger=logger,
            test_in_train=False,
            reward_metric=reward_metric
        )

        save_best_fn(self.policy, suffix="final")

        return result, [ self.policy.policies[self.agents[agent_id]] for agent_id in range(CONFIG[CFG_ENV]['n_total_agents']) ]

    # Generic setters for vars used during execution (i.e. non-cli args)
    def set_world_seed(
        self,
        seed: int
    ) -> None: 
        self.world_seed = seed

    def set_noise_seed(
        self,
        seed: int
    ) -> None:
        self.noise_seed = seed

    def set_max_cycles(
        self,
        cycles: int
    ) -> None:
        self.max_cycles = cycles

    def set_initial_state(
        self,
        initial_state: np.ndarray,
    ) -> None:
        self.initial_state = initial_state

    # Run n simulations of the trained policies
    def run_simulator(
        self,
        n_episodes: Optional[int] = 1,
        epsilon: Optional[float] = None,
        target_epsilon: Optional[float] = None,
    ) -> None:
        if type(self.initial_state) is np.ndarray:
            env = self.get_env(initial_state=np.copy(self.initial_state))
        else:
            env = self.get_env()
        env = DummyVectorEnv([lambda: env])
        self.policy.eval()
        # Set all agent policies to test
        epsilon = CONFIG[CFG_SIM]['policy']['eps_test'] if not epsilon else epsilon
        # TODO Major bodge. Needs refactoring properly.
        target_epsilon = CONFIG[CFG_SIM]['policy']['eps_test']
        for agent_idx in range(CONFIG[CFG_ENV]['n_total_agents']):
            # TODO Refactor this!
            # Set only ego agent policy to target epsilon
            self.policy.policies[self.agents[agent_idx]].set_eps(epsilon)        
        test_buffer = VectorReplayBuffer(CONFIG[CFG_SIM]['buffer_size'], 1)
        collector = Collector(self.policy, env, test_buffer, exploration_noise=True)
        result = collector.collect(n_episode=n_episodes, render=CONFIG[CFG_SIM]['policy']['render'])
        rews, lens = result["rews"], result["lens"]
        return test_buffer

    def output_observations(
        self,
        buffer: VectorReplayBuffer,
        prefix: Optional[bool] = False,
        n_episodes: Optional[int] = 1,
        subfolder: Optional[str] = None,
        fn_suffix: Optional[str] = ""
    ) -> Tuple[np.ndarray]:
        output_path = os.path.join(CONFIG[CFG_PATH]['data'], subfolder) if subfolder else CONFIG[CFG_PATH]['data']
        if not os.path.exists(output_path): os.makedirs(output_path)
        states = np.zeros(shape=(n_episodes * self.max_cycles, self.state_space_dimen))
        rewards = np.zeros(shape=(n_episodes * self.max_cycles, self.action_dimen * CONFIG[CFG_ENV]['n_total_agents']))
        for i in range(0, (self.max_cycles * n_episodes * CONFIG[CFG_ENV]['n_total_agents']), CONFIG[CFG_ENV]['n_total_agents']):
            true_idx = i // CONFIG[CFG_ENV]['n_total_agents']
            agent_pos_vels = []
            rewards[true_idx] = buffer.get(i + CONFIG[CFG_ENV]['n_total_agents'] - 1, key='rew')
            # print(rewards[true_idx])
            landmark_pos = []
            for k in range(CONFIG[CFG_ENV]['n_total_agents']):
                batch = buffer.get(i + k, key="obs")
                # print(f"Action: {buffer.get(i + k, key='act')}")
                agent_obs = batch['obs']
                # agent_action = buffer.get(i + k, key="act")
                if batch['agent_id'] is None:
                    print(f"[WARN] More observations requested than episodes in the buffer. Returning {i} observations.")
                    break
                    # Not sure if I want to raise an exception here, but it does not matter too much for now.
                    # raise Exception("There are not enough episodes in the buffer for the number of observations requested.")
                if k == 0:
                    # Get landmark pos
                    # TODO: Make this more robust and fully document state space somewhere
                    for l in range(0, CONFIG[CFG_ENV]['n_landmarks'] * 2, 2):
                        # TODO This is rather horrible. Need to fix this.
                        landmark_idx = 4 + ((CONFIG[CFG_ENV]['n_total_agents'] - 1) * 2)
                        landmark = agent_obs[(landmark_idx + l):(landmark_idx + l + 2)]
                        # Get back absolute position
                        landmark_pos.append(agent_obs[:2] + landmark)
                agent_pos_vels.append(agent_obs[:4])
            state_vec = np.concatenate(agent_pos_vels + landmark_pos)
            states[true_idx] = state_vec
        if fn_suffix != "": fn_suffix = f"_{fn_suffix}"
        # If we're in two-step mode and this isn't the prefix, adjust the state to account for
        # additional step added in trajectory generation
        if not prefix and CONFIG[CFG_DATA]['prefix_len']:
            states = states[1:]
            rewards = rewards[:-1]
        # Only if we are the prefix, don't 
        np.save(os.path.join(output_path, f"state{fn_suffix}"), states)
        np.save(os.path.join(output_path, f"rewards{fn_suffix}"), rewards)
        # Return state and actions as tuple
        return (states, rewards)

    def generate_trajectories(self) -> None:
        self.set_max_cycles(CONFIG[CFG_DATA]['trajectory_len'])
        params = [CONFIG[CFG_DATA]['n_states'], CONFIG[CFG_DATA]['n_suffixes'], CONFIG[CFG_DATA]['trajectory_len'], CONFIG[CFG_ENV]['n_total_agents']]
        np.save(os.path.join(CONFIG[CFG_PATH]['data'], f"params.npy"), np.array(params))
        # Save target epsilons to disk
        if CONFIG[CFG_DATA]['eps_target']:
            np.save(os.path.join(CONFIG[CFG_PATH]['data'], f"target_epsilon.npy"), np.array(CONFIG[CFG_DATA]['eps_target']))
        seeds_fpath = os.path.join(CONFIG[CFG_PATH]['data'], f"seeds.npy")
        seeds = self.world_rng.choice(99999, CONFIG[CFG_DATA]['n_states'], replace=False)
        # Check if seeds already exists as we need to update it if so
        # TODO: Implement more robust function here to check if the corresponding folders actually exist
        if os.path.exists(seeds_fpath):
            old_seeds = np.load(seeds_fpath)
            #new_seeds = np.unique(np.concatenate((old_seeds, seeds), axis=0))
            #np.save(seeds_fpath, new_seeds)
        else:
            np.save(seeds_fpath, seeds)
        seeds = list(np.load(seeds_fpath))
        if CONFIG[CFG_SIM]['generator']['parallel']:
            print(f"[INFO] Launching {CONFIG[CFG_SIM]['generator']['n_threads']} generator threads...")
            self._gen_traj_parallel(seeds)
        else:
            # Generate trajectories one by one.
            for seed in seeds:
                self._gen_traj_seq(seed)

    # Generates trajectories sequentially
    def _gen_traj_seq(
        self,
        seed: int,
        job_num: Optional[int] = 0
    ) -> None:
        # FIXME: This requires serious refactoring
        if job_num: print(f"[INFO] Starting job {job_num}...")
        self.set_world_seed(seed + job_num)
        # Generate prefix
        initial_state = None
        if CONFIG[CFG_DATA]['prefix_len']:
            initial_path = os.path.join(CONFIG[CFG_PATH]['data'], f"seed_{seed}", "state_initial.npy")
            self.set_max_cycles(CONFIG[CFG_DATA]['prefix_len'])
            prefix_buffer = self.run_simulator()
            # Save prefix states and save to disk
            states, _ = self.output_observations(prefix_buffer, prefix=True, subfolder=f"seed_{seed}", fn_suffix=f"prefix")
            initial_state = states[-1]
            self.set_initial_state(initial_state)
            np.save(initial_path, initial_state)
            # Add 1 to allow for initial step
            self.set_max_cycles(CONFIG[CFG_DATA]['suffix_len'] + 1)
            # Reset noise seed for subsequent env runs
            self.set_noise_seed(None)
        for t in range(CONFIG[CFG_DATA]['n_suffixes']):
            # Generate behavioural
            buffer = self.run_simulator()
            # TODO Bodge again. Fn_suffix manually changed to omit '_b'
            self.output_observations(buffer, prefix=False, subfolder=f"seed_{seed}", fn_suffix=f"traj_{t}")

    # Generates trajectories in parallel with a given max number of threads
    def _gen_traj_parallel(
        self,
        seeds: list
    ) -> None:
        mp.set_start_method('spawn')
        pool = Pool(processes=CONFIG[CFG_SIM]['generator']['n_threads'], maxtasksperchild=10)
        job_num = 0
        # Keep scheduling jobs so long as we have threads free and more seeds
        # to evaluate
        while(seeds):
            job_num += 1
            seed = seeds.pop()
            pool.apply_async(self._gen_traj_seq, (seed, job_num,))
        
        pool.close()  # Done adding tasks.
        pool.join()  # Wait for all tasks to complete.
