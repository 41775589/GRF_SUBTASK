import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to promote ball control, strategic play, and effective ball distribution."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        self.reward_scale = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        # Custom state representation for serialization
        to_pickle['position_rewards'] = self.position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle['position_rewards']
        return from_pickle

    def reward(self, reward):
        # Accessing the environment's current observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "ball_control_reward": [0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Apply an additional reward for maintaining ball possession under pressure
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # The player is actively controlling the ball
                components["ball_control_reward"][rew_index] = self.reward_scale
                reward[rew_index] += components["ball_control_reward"][rew_index]

                # Reward distribution of the ball effectively across the field
                player_pos = o['left_team'][o['active']]
                for teammate_pos in o['left_team']:
                    if np.linalg.norm(player_pos - teammate_pos) > 0.3:  # Encourage passes over distance
                        reward[rew_index] += self.reward_scale / 2

            # Discourage keeping the ball in highly defensive positions
            if o['ball_owned_team'] == 0 and o['active'] in o['left_team']:
                if np.abs(o['ball'][0]) < 0.2:  # Near the own goal
                    reward[rew_index] -= self.reward_scale * 2

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Accumulate rewards in the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Recording the counts of sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
