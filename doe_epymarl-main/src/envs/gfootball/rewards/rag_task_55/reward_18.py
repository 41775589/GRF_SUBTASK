import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This reward wrapper specifically focuses on rewarding agents based on
    their ability to perform sliding and standing tackles effectively,
    without committing fouls, during various gameplay scenarios."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize a counter for the number of successful tackles
        self.successful_tackles = 0
        # Coefficients for computing the tackle rewards
        self.tackle_reward_coefficient = 0.5
        # Initialize counter for fouls committed during tackles
        self.fouls_committed = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """Resets the environment and reward-related metrics."""
        self.successful_tackles = 0
        self.fouls_committed = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Custom reward logic focusing on the quality of tackles performed by agents."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Prepare reward components
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for tackles without fouls
            if o['game_mode'] == 3 or o['game_mode'] == 6:  # Assuming game_mode 3 is fouls and 6 is tackles
                if o['left_team_yellow_card'][o['active']] or o['right_team_yellow_card'][o['active']]:
                    self.fouls_committed += 1
                    components["foul_penalty"][rew_index] = -0.1
                else:
                    self.successful_tackles += 1
                    components["tackle_reward"][rew_index] = self.tackle_reward_coefficient

            # Update the reward for this agent
            reward[rew_index] += components["tackle_reward"][rew_index] + components["foul_penalty"][rew_index]

        return reward, components
    
    def step(self, action):
        """Performs a step in the environment, adjusts rewards, and updates observations."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
            for i, count in enumerate(self.sticky_actions_counter):
                info[f"sticky_actions_{i}"] = count
        return observation, reward, done, info
