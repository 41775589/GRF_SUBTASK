import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful sliding tackles, focusing on the timing aspect.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_sliding_tackles = 0
        self.successful_tackles = 0
        self.tackle_reward = 2.0  # Large reward for successful tackles

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_sliding_tackles = 0
        self.successful_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'total_sliding_tackles': self.total_sliding_tackles,
            'successful_tackles': self.successful_tackles
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.total_sliding_tackles = from_pickle['CheckpointRewardWrapper']['total_sliding_tackles']
        self.successful_tackles = from_pickle['CheckpointRewardWrapper']['successful_tackles']
        return from_pickle

    def reward(self, reward):
        # Original reward passed from the game
        original_reward = reward.copy()

        # New reward component
        tackle_success_reward = [0.0] * len(reward)
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {'base_score_reward': original_reward, 'tackle_success_reward': tackle_success_reward}

        for rew_index, o in enumerate(observation):
            if self.action_was_tackle(o) and self.tackle_was_successful(o):
                tackle_success_reward[rew_index] = self.tackle_reward
                reward[rew_index] += tackle_success_reward[rew_index]
                self.successful_tackles += 1

        # Reward components for debug
        components = {
            'base_score_reward': original_reward,
            'tackle_success_reward': tackle_success_reward
        }

        return reward, components

    def action_was_tackle(self, obs):
        # Assuming observation includes the last action made; 
        # Use action indices that correspond to tackle moves
        return obs.get('last_action', -1) == 8

    def tackle_was_successful(self, obs):
        # A simplistic function to decide if a tackle was successful,
        # Could be based on the change in ball possession or other game metrics
        return obs.get('ball_owned_team', -1) == 0 # Assuming 0 is our team

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += int(act)
        return observation, reward, done, info
