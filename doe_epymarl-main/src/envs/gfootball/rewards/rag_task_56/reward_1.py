import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive tasks such as improving shot-stopping for 
    goalkeepers and enhancing defenders' tackling and ball-retention abilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_performance = 0.05
        self.defender_efficiency = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_performance": [0.0] * len(reward),
                      "defender_efficiency": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Goalkeeper enhancement for shot-stopping
            if o['right_team_roles'][o['active']] == 0:  # assuming role 0 is goalkeeper
                # Ideally, we should check for shot stopping events or saves, but this is a proxy suggestion
                if o['ball_owned_team'] == 0:
                    components["goalkeeper_performance"][rew_index] = self.goalkeeper_performance
                    reward[rew_index] += components["goalkeeper_performance"][rew_index]

            # Defender tackling and ball control
            if o['right_team_roles'][o['active']] in {1, 2, 3, 4}:  # assuming roles {1, 2, 3, 4} are defenders
                # Another proxy: ball control within a defensive area of the field
                if o['ball'][0] < -0.5 and o['ball_owned_team'] == 0:
                    components["defender_efficiency"][rew_index] = self.defender_efficiency
                    reward[rew_index] += components["defender_efficiency"][rew_index]
        
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
