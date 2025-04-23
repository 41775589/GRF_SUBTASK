import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that awards agents based on their ability to effectively master high passes 
    and position themselves as wide midfielders, supporting lateral transitions and stretching 
    the defense.
    """
    def __init__(self, env):
        super().__init__(env)
        # Parameters to adjust the impact of different reward components
        self.high_pass_reward = 0.2
        self.positioning_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_bonus": [0.0] * len(reward),
                      "positioning_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # High pass action: assuming high pass corresponds to action Top in sticky actions (index 2)
            if o['sticky_actions'][2] == 1:
                components['high_pass_bonus'][rew_index] = self.high_pass_reward
                reward[rew_index] += self.high_pass_reward

            # Positioning reward for wide midfielders (getting close to sidelines)
            player_x, player_y = o['left_team'][o['active']]
            sideline_dist = min(abs(player_y - 0.42), abs(player_y + 0.42))
            
            # Encourage players to be near sidelines but not too close (thresholds can be adjusted)
            if sideline_dist <= 0.1:
                components['positioning_bonus'][rew_index] = self.positioning_reward
                reward[rew_index] += self.positioning_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
