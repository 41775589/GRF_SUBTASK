import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage offensive plays through precision-based finishing and fast-paced maneuvers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._speed_bonus = 0.05
        self._precision_bonus = 0.1
        self._goal_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Get the original observation
        components = {"base_score_reward": reward.copy(),
                      "speed_bonus": [0.0] * len(reward),
                      "precision_bonus": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'score' in o:
                original_goal_reward = reward[rew_index]
                # Check if a goal was scored.
                if o['score'][0] > 0:
                    components["precision_bonus"][rew_index] = self._precision_bonus * original_goal_reward
                reward[rew_index] = components["base_score_reward"][rew_index] + components["precision_bonus"][rew_index]

            # Speed bonus for fast actions; assessing using ball velocity thresholds
            if 'ball_direction' in o:
                speed = np.linalg.norm(o['ball_direction'])
                if speed > 0.1:  # Example threshold for what's considered 'fast-paced'
                    components["speed_bonus"][rew_index] = self._speed_bonus * speed
                reward[rew_index] += components["speed_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
