import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages agents to master dribbling and sprinting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_progression = {}
        self.current_sprint_usage = {}
        self.sprint_bonus = 0.05
        self.dribble_bonus = 0.1
        self.max_sprint_steps = 20
        self.max_dribble_steps = 50

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_progression = {}
        self.current_sprint_usage = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_progression'] = self.dribble_progression
        to_pickle['current_sprint_usage'] = self.current_sprint_usage
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_progression = from_pickle['dribble_progression']
        self.current_sprint_usage = from_pickle['current_sprint_usage']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "dribble_reward": [0.0] * len(reward), 
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player = o['active']
            sticky_actions = o['sticky_actions']

            # Bonus for using dribble
            dribble_active = sticky_actions[9]  # position of dribble action in sticky actions
            if dribble_active:
                self.dribble_progression[rew_index] = self.dribble_progression.get(rew_index, 0) + 1
                if self.dribble_progression[rew_index] < self.max_dribble_steps:
                    components["dribble_reward"][rew_index] = self.dribble_bonus
            
            # Bonus for using sprint effectively
            sprint_active = sticky_actions[8]  # position of sprint action in sticky actions
            if sprint_active:
                self.current_sprint_usage[rew_index] = self.current_sprint_usage.get(rew_index, 0) + 1
                if self.current_sprint_usage[rew_index] < self.max_sprint_steps:
                    components["sprint_reward"][rew_index] = self.sprint_bonus

            reward[rew_index] += (components["dribble_reward"][rew_index] +
                                  components["sprint_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
