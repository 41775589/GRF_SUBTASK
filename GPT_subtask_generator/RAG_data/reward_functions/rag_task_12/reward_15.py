import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards based on player actions and ball control in football environment."""
    
    def __init__(self, env):
        super().__init__(env)
        self.dribble_reward = 0.05
        self.high_pass_reward = 0.1
        self.long_pass_reward = 0.1
        self.sprint_reward = 0.02
        self.stop_sprint_reward = 0.02
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward),
                      "stop_sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Dribble action check
            if o['sticky_actions'][9] == 1:  # action_dribble is active
                reward[rew_index] += self.dribble_reward
                components["dribble_reward"][rew_index] += self.dribble_reward
            
            # High pass or long pass action, depending on ball's direction y-coord
            if o['ball_direction'][1] > 0.1:  # Positive y direction implies forward pass
                if o['sticky_actions'][8] == 1:  # action_sprint is active
                    reward[rew_index] += self.high_pass_reward
                    components["high_pass_reward"][rew_index] += self.high_pass_reward
                else:
                    reward[rew_index] += self.long_pass_reward
                    components["long_pass_reward"][rew_index] += self.long_pass_reward
            
            # Sprint and stop sprint rewards
            if o['sticky_actions'][8] == 1:  # action_sprint is active
                reward[rew_index] += self.sprint_reward
                components["sprint_reward"][rew_index] += self.sprint_reward
            else:
                reward[rew_index] += self.stop_sprint_reward
                components["stop_sprint_reward"][rew_index] += self.stop_sprint_reward

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
