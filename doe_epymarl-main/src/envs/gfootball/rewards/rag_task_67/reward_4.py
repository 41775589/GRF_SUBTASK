import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for transition skills like Short Pass, Long Pass, and Dribble."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player = o['active']

            # Reward for successfully using dribble or pass in a pressure situation
            if o['sticky_actions'][8] or o['sticky_actions'][9]:  # Dribble or Sprint
                # Check if the player with the ball is dribbling or sprinting effectively under pressure
                if o['ball_owned_player'] == active_player:
                    opponents = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
                    ball_owning_player_pos = o['left_team'][active_player] if o['ball_owned_team'] == 0 else o['right_team'][active_player]
                    # Calculate the pressure by the proximity of the nearest opponent
                    distances = np.linalg.norm(opponents - ball_owning_player_pos, axis=1)
                    pressure = np.any(distances < 0.1)  # High pressure if an opponent is very close
                    
                    if pressure:
                        components["pass_dribble_reward"][rew_index] = 0.1  # Rewarding the player
                        reward[rew_index] += components["pass_dribble_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
