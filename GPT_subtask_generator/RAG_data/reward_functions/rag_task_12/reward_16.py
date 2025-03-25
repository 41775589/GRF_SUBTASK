import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focusing on the transition plays of midfielders/defenders."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for specific actions
        self.high_pass_reward = 0.1
        self.long_pass_reward = 0.1
        self.dribble_reward = 0.05
        self.sprint_reward = 0.03
        self.stop_sprint_reward = 0.03

    def reset(self):
        """Reset sticky actions counter on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Compute the auxiliary rewards based on player's actions and game dynamics."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": 0.0,
                      "long_pass_reward": 0.0,
                      "dribble_reward": 0.0,
                      "sprint_reward": 0.0,
                      "stop_sprint_reward": 0.0}

        if observation is None:
            return reward, components
        
        for idx in range(len(reward)):
            o = observation[idx]
            
            # Checking relevant actions taken
            # Sprint action
            if o['sticky_actions'][8] == 1:
                reward[idx] += self.sprint_reward
                components["sprint_reward"] += self.sprint_reward

            # Stop Sprint action
            if o['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] > 0:
                reward[idx] += self.stop_sprint_reward
                components["stop_sprint_reward"] += self.stop_sprint_reward
            
            # Dribble under pressure
            if o['sticky_actions'][9] == 1:
                reward[idx] += self.dribble_reward
                components["dribble_reward"] += self.dribble_reward

            # High Pass or Long Pass detection using action not directly available,
            # typically should be checked through mode or result in game state
            if 'game_mode' in o and o['game_mode'] == 3:  # for e.g., FreeKick as a proxy
                reward[idx] += self.high_pass_reward
                components["high_pass_reward"] += self.high_pass_reward
            
            # Updating counters
            self.sticky_actions_counter = o['sticky_actions'].copy()

        return reward, components

    def step(self, action):
        """Execute environment step and compute reward modifications."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
