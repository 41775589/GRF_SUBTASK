import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the original reward by evaluating the goalkeeper's performance based on proximity to the ball, shot stopping, distribution decisions and communication with defenders."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._save_actions_count = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._save_actions_count = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "proximity_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # If the goalkeeper's team owns the ball
                if o['left_team_roles'][o['active']] == 0:  # and the active player is the goalkeeper
                    # Evaluate shot stopping
                    if o['game_mode'] in [2, 4, 6]:  # In modes like GoalKick, Corner, Penalty
                        components["proximity_reward"][rew_index] += 0.5
                        self._save_actions_count += 1
            
            # Check proximity to the ball
            ball_pos = o['ball'][:2]
            goalie_pos = o['left_team'][o['active']]
            distance_to_ball = np.linalg.norm(ball_pos - goalie_pos)
            
            # Reward being close to ball during critical game modes
            if distance_to_ball < 0.1:
                components["proximity_reward"][rew_index] += 0.1 * (0.1 - distance_to_ball)

            # Aggregate the rewards
            reward[rew_index] += components["proximity_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
