import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

from vrcp_reg_config import *
# from ma_copp.util.eps import apply_eps

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        seed_w,
        seed_n,
        local_ratio=0.5,
        max_cycles=10,
        continuous_actions=False,
        initial_state=None,
        target_env=False
    ):
        EzPickle.__init__(
            self, CONFIG[CFG_ENV]['n_agents'], local_ratio, max_cycles, continuous_actions, CONFIG[CFG_SIM]['render_mode']
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario(seed_w, seed_n, initial_state=initial_state)
        self.world = scenario.make_world()
        super().__init__(
            scenario=scenario,
            world=self.world,
            render_mode=CONFIG[CFG_SIM]['render_mode'],
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v2"
        self.target_env = target_env

    # NOTE There might be a nicer way of implementing this, but I'm hooking
    # it just before we actually set the actions and take the world step.
    # Handles the rotation bias
    def _set_action(self, action, agent, action_space, time=None):
        # Only if we are operating the target policies
        # if self.target_env:
        #     # Only apply epsilon greedy for ego agent
        #     # TODO Implement list of ego agents
        #     agent_idx = self.world.agents.index(agent)
        #     if agent_idx == 0:
        #         action = apply_eps(action)
        super()._set_action(action, agent, action_space, time=time)
        


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):

    def __init__(self, world_seed, noise_seed, initial_state=None):
        self.world_seed = world_seed
        self.noise_seed = noise_seed
        self.noisy = CONFIG[CFG_RAND]['sim']['noisy']
        self.initial_state = initial_state
        if self.noisy:
            self.noise_rng = np.random.default_rng(seed=self.noise_seed)
            self.noise_variation = CONFIG[CFG_RAND]['sim']['noise_var']
            self.noise_func = getattr(self.noise_rng, CONFIG[CFG_RAND]['sim']['noise_func'])
        self.world_rng = np.random.default_rng(seed=self.world_seed)
        self.has_reset = True
        self.agent_initial = None    

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = CONFIG[CFG_ENV]['n_agents']
        num_landmarks = CONFIG[CFG_ENV]['n_agents']
        world.num_agents = num_agents
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # TODO Change this to a CLI arg
            agent.u_noise = 0.01
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def random_reset(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.world_rng.uniform(-5, +5, world.dim_p)
            # TODO Make this configurable
            #agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel = self.world_rng.uniform(-0.5, +0.5, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.world_rng.uniform(-2.5, +2.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def load_initial_state(self, world):
        #[ pos, vel, pos, vel, pos, vel, landmark1, landmark2, landmark3 ]
        agent_bound = int(len(world.agents) * 4)
        # Agent pos
        for i in range(0, agent_bound, 4):
            agent = world.agents[i // 4]
            agent.state.p_pos = self.initial_state[i:(i + 2)]
            agent.state.p_vel = self.initial_state[(i + 2):(i + 4)]
            agent.state.c = np.zeros(world.dim_c)
        # Landmark pos
        for i in range(agent_bound, len(self.initial_state), 2):
            idx = (i - (agent_bound)) // 2
            landmark = world.landmarks[idx]
            landmark_pos = self.initial_state[i:(i + 2)]
            landmark.state.p_pos = landmark_pos
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reset_world(self, world, np_random):
        self.has_reset = True
        self.agent_initial = [ True ] * world.num_agents
        if type(self.initial_state) is np.ndarray:
            self.load_initial_state(world)
        else:
            self.random_reset(world)
        

    # def benchmark_data(self, agent, world):
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for lm in world.landmarks:
    #         dists = [
    #             np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
    #             for a in world.agents
    #         ]
    #         min_dists += min(dists)
    #         rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for other in world.agents:
                if agent is not other and self.is_collision(agent, other):
                    rew -= 1
        return rew

    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):

        agent_idx = world.agents.index(agent)

        # Applies noise according to a specified distribution and ensuring resulting observation is valid
        # For now, only apply noise to agents observations of other agents, not its own position/velocity
        # TODO: Refactor this into its own function so that it can be applied to any PettingZoo env
        def apply_noise(clean_data):
            # If we have only just reset, don't add initial noise to observation as we
            # want to continue from the previous observation state
            if not self.noisy or self.agent_initial[agent_idx]:
                return clean_data
            else:
                perturbed_data = np.full(clean_data.shape, np.inf)
                # TODO: Make the world dimensions configurable
                # FIXME: Make robust against noise var exceeding the world dimensions (causing a inf. loop)
                # Retry generating a random number if the resulting observation goes beyond the world dimensions
                #while not np.all((perturbed_data >= -2) & (perturbed_data <= 2)):
                if CONFIG[CFG_RAND]['sim']['noise_func'] == "normal":
                    # Multiply each element by independent Gaussian noise
                    noise = self.noise_rng.normal(loc=0, scale=self.noise_variation, size=clean_data.shape) + 1
                    perturbed_data = clean_data * noise
                else:
                    noise = self.noise_func(-self.noise_variation, self.noise_variation, clean_data.shape)
                    perturbed_data = clean_data + noise
                return perturbed_data

        # get positions of all entities in this agent's reference frame
        entity_pos = np.array([ entity.state.p_pos for entity in world.landmarks ]).flatten()
        
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(apply_noise(other.state.p_pos))

        other_pos = np.array(other_pos).flatten()

        obs = np.concatenate(
                (agent.state.p_pos, agent.state.p_vel, other_pos, entity_pos)
            ).flatten()
        
        self.agent_initial[agent_idx] = False

        return obs
