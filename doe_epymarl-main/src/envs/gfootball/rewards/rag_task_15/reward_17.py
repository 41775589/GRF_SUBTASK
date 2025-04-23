import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on long pass precision and dynamics."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_pass_bonus = 0.5  # Reward bonus for successful long passes
        self.pass_accuracy_requirement = 0.8  # Minimum accuracy required to get bonus
        self.min_pass_distance = 0.5  # Minimum distance considered a 'long pass'
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['long_pass_data'] = {"long_pass_bonus": self.long_pass_bonus,
                                       "pass_accuracy_requirement": self.pass_accuracy_requirement,
                                       "min_pass_distance": self.min_pass_distance}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        long_pass_data = from_pickle['long_pass_data']
        self.long_pass_bonus = long_pass_data['long_pass_bonus']
        self.pass_accuracy_requirement = long_pass_data['pass_accuracy_requirement']
        self.min_pass_distance = long_pass_data['min_pass_distance']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            
            # Calculate distance of the ball pass
            if 'ball_direction' in o:
                pass_distance = np.linalg.norm(o['ball_direction'][:2])  # Considering only x, y
                
                # Check if it's a long pass
                if pass_distance > self.min_pass_distance:
                    if o['ball_owned_team'] == o['active'] and 'game_mode' in o:
                        # Assuming a successful control of the ball after a long pass
                        if o['game_mode'] == 0 and np.random.random() < self.pass_accuracy_requirement:
                            additional_reward = self.long_pass_bonus
                            components['long_pass_precision_reward'][idx] = additional_reward
                            reward[idx] += additional_reward
                
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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
