import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for dribbling maneuvers combined with dynamic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_reward = 0.1
        self.position_change_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Initialize the components of the reward (base score and new elements for dribbling/position change)
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "position_change_reward": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        # Loop through observations to apply the enhanced rewards
        for rew_index, obs in enumerate(observation):
            is_dribbling = obs['sticky_actions'][9] == 1 # Dribble action is indexed at 9
            if is_dribbling:
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Reward for significant position change, considering x-direction (left-right field movement)
            if 'previous_position' in obs:
                previous_x = obs['previous_position'][0]
                current_x = obs['left_team'][obs['active']][0] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']][0]
                if abs(current_x - previous_x) > 0.05:  # Threshold for significant position change
                    components["position_change_reward"][rew_index] = self.position_change_reward
                    reward[rew_index] += components["position_change_reward"][rew_index]
                obs['previous_position'][0] = current_x  # Update previous x position

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
