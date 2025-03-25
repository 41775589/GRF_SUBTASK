import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing precise high passes in the football environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_precision_threshold = 0.1
        self.pass_height_threshold = 0.15
        self.pass_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Nothing specific to restore for this wrapper
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_skill_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Iterate through each player's observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Ensuring we have the right conditions: ball ownership and action checks
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and 
                'action' in o and o['action'] == 'high_pass' and
                'ball_direction' in o and o['ball_rotation' in o]):
                
                ball_height = o['ball'][2]
                ball_speed_z = o['ball_direction'][2]
                
                # Check if pass is high and with controlled precision and power
                if ball_height > self.pass_height_threshold and abs(ball_speed_z) < self.pass_precision_threshold:
                    components["pass_skill_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_skill_reward"][rew_index]

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
