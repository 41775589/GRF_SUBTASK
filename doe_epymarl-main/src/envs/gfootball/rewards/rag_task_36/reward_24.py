import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes dribbling and dynamic positioning rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_coefficient = 0.05
        self.position_change_coefficient = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "position_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            previous_position = player_obs.get('previous_position', player_obs['right_team'] if player_obs['ball_owned_team'] == 1 else player_obs['left_team'])
            current_position = player_obs['right_team'] if player_obs['ball_owned_team'] == 1 else player_obs['left_team']

            # Reward for dribbling
            if player_obs['sticky_actions'][9]:  # Assuming 9 is the dribble action index
                components["dribble_reward"][rew_index] = self.dribble_coefficient

            # Reward for dynamic positioning (fluid transitions)
            position_change = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            components["position_change_reward"][rew_index] = self.position_change_coefficient * position_change
            
            # Update rewards
            reward[rew_index] += components["dribble_reward"][rew_index] + components["position_change_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
