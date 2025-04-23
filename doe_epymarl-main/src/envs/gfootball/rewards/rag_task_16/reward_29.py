import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for achieving key skills in executing high passes with precision.
    This includes evaluating the trajectory accuracy, power assessment, and the situation
    where high passes are advantageous.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.high_pass_reward = 0.2
        self.accuracy_threshold = 0.1    # Threshold for considering the pass as precise.
        self.power_coefficient = 0.3      # Coefficient to adjust power relevance in reward.
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
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_direction' in o:
                z_direction = o['ball_direction'][2]  # Z component direction of the ball
                power = np.linalg.norm(o['ball_direction'])
                
                # Check for high pass situation: High Z direction and reasonable power
                if z_direction > self.accuracy_threshold and power > 0.5:
                    adjustment = min(1.0, z_direction * self.power_coefficient) * self.high_pass_reward
                    components["high_pass_reward"][rew_index] = adjustment
                    reward[rew_index] += adjustment
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
