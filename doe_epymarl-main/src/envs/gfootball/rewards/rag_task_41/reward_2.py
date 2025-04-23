import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense attacking skill-focused reward, 
    promoting finishing accuracy and creative offensive play 
    under match-like pressures and defensive setups."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._accuracy_reward = 0.2
        self._creativity_reward = 0.1
        self._pressure_handling_reward = 0.1

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
        components = {
            "base_score_reward": reward.copy(),
            "accuracy_reward": [0.0] * len(reward),
            "creativity_reward": [0.0] * len(reward),
            "pressure_handling_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # If the ball is owned by the right team (attacking team)
                if o['score'][1] > o['score'][0]:  # Reward if right team scores
                    components["accuracy_reward"][rew_index] = self._accuracy_reward
                    reward[rew_index] += components["accuracy_reward"][rew_index]
                
                if np.any(np.diff(o['right_team_direction'], axis=0)):  # Player creativity in moving
                    components["creativity_reward"][rew_index] = self._creativity_reward
                    reward[rew_index] += components["creativity_reward"][rew_index]
                    
                if o['game_mode'] in (2, 3, 4, 5, 6):  # Handling pressure in diverse game situations
                    components["pressure_handling_reward"][rew_index] = self._pressure_handling_reward
                    reward[rew_index] += components["pressure_handling_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
