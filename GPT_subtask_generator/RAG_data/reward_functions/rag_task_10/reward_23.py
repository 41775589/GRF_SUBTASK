import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.3  # Reward for intercepting the ball
        self.positioning_reward = 0.1   # Reward for optimal positioning
        self.defensive_actions = {6, 7, 8}  # indices for defensive actions: Stop, Slide, 
        self.good_positions = [(0.2, 0.0), (-0.2, 0.0)]  # Optimal defensive positions on field

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 1:
                # Interception logic, reinforced when the player intercepts the ball
                components["interception_reward"][rew_index] = self.interception_reward
                reward[rew_index] += components["interception_reward"][rew_index]
            
            # Reward defensive positioning on the field
            player_pos = o['right_team' if o['active'] in range(5) else 'left_team'][o['active']]
            for good_pos in self.good_positions:
                dist = np.linalg.norm(np.array(player_pos) - np.array(good_pos))
                if dist < 0.1:  # Threshold to consider good positioning
                    components["positioning_reward"][rew_index] = self.positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]
                    break
        
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
