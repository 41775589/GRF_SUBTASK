import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific defensive training rewards to the environment."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize a counter for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize tracking of tackles and goalkeeper saves
        self.tackles_saved = 0
        self.goalkeeper_saves = 0
        self.goalkeeper_passes = 0

    def reset(self):
        """Reset the counters and environment states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_saved = 0
        self.goalkeeper_saves = 0
        self.goalkeeper_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to preserve current training progress."""
        to_pickle['tackle_and_save_counts'] = (self.tackles_saved, self.goalkeeper_saves, self.goalkeeper_passes)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from loaded progress."""
        from_pickle = self.env.set_state(state)
        self.tackles_saved, self.goalkeeper_saves, self.goalkeeper_passes = from_pickle['tackle_and_save_counts']
        return from_pickle

    def reward(self, reward):
        """Customize the reward based on defensive actions."""
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy()}
        
        for idx, o in enumerate(observation):
            if o['left_team_roles'][o['active']] == 0:  # Goalkeeper
                if o['ball_owned_player'] == o['active']:
                    if self.sticky_actions_counter[8] > 0:  # Sprint, indicative of quick movement/initiation
                        self.goalkeeper_passes += 1
                        components.setdefault('goalkeeper_pass_reward', []).append(0.2)
                        reward[idx] += 0.2
                    self.goalkeeper_saves += 1
                    components.setdefault('goalkeeper_save_reward', []).append(0.3)
                    reward[idx] += 0.3
            # Check for tackles (role is defensive and a turnover happens from team 1 to 0)
            elif o['left_team_roles'][o['active']] in [1, 2, 3, 4] and o['ball_owned_team'] == 0:
                self.tackles_saved += 1
                components.setdefault('tackle_reward', []).append(0.5)
                reward[idx] += 0.5
        
        return reward, components

    def step(self, action):
        """Apply actions, update environment, and modify the reward based on defensive performance."""
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
        return observation, reward, done, info
