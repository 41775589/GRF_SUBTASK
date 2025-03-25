import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a focused defensive reward for successful sliding tackles and more importantly 
    the precision and timing in initiating them under high-pressure situations.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize all agent-specific counters for tracking sliding tackles rewards
        self.sliding_tackle_counter = np.zeros(10, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for precise and timely tackles under pressure
        self._tackle_reward = 0.5
        self._pressure_tackle_bonus = 0.3

    def reset(self):
        # Reset all counters upon starting a new environment episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Keep the original reward structure
        base_reward = reward
        tackle_reward = [0.0] * len(reward)
        
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": tackle_reward}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 6:  # Assume mode 6 is intense or pressure mode requiring defenses
                if 'ball_owned_team' not in o:
                    continue
                if o['ball_owned_team'] == 1 and o.get('action', None) == 'slide_tackle':
                    # Only consider tackles from our controlled team, team ID 1 in this context
                    components["tackle_reward"][rew_index] += self._tackle_reward
                    if self.sliding_tackle_counter[rew_index] == 0:
                        # First successful tackle under pressure gets a bonus
                        components["tackle_reward"][rew_index] += self._pressure_tackle_bonus
                    self.sliding_tackle_counter[rew_index] += 1
                    reward[rew_index] += components["tackle_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        # Predefined from the given environment; adds reward components to steps
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
