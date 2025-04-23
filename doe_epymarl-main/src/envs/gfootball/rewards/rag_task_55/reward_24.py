import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive tactics by rewarding successful tackles without causing fouls."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = 0
        self.tackles_fouled = 0
        self.reward_for_tackle = 0.1
        self.penalty_for_foul = -0.05

    def reset(self):
        """Reset the environment and statistics for tackles."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = 0
        self.tackles_fouled = 0
        return self.env.reset()

    def reward(self, reward):
        """Calculate custom reward based on the quality of tackles and avoidance of fouls."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "foul_penalty": [0.0] * len(reward)
        }
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'game_mode' in o:
                if o['game_mode'] == 6:  # Enum for Penalty mode which might indiciate fouls recently.
                    components['foul_penalty'][rew_index] = self.penalty_for_foul
                    reward[rew_index] += components['foul_penalty'][rew_index]
                    self.tackles_fouled += 1
                elif o['game_mode'] == 3:  # Enum for FreeKick mode, successful tackle without fouling.
                    components['tackle_reward'][rew_index] = self.reward_for_tackle
                    reward[rew_index] += components['tackle_reward'][rew_index]
                    self.tackles_successful += 1

        return reward, components

    def step(self, action):
        """Perform one timestep within the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_flag
            
        return observation, reward, done, info
