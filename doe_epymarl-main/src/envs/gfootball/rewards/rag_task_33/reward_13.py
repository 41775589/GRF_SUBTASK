import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward function for encouraging long-range shots from outside the penalty box.
    It rewards the agent for holding possession and attempting distant shots towards the goal.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Penalty box boundary, distance from the center (goal line to penalty box line is roughly 0.18 on x-axis)
        self.penalty_box_x_threshold = 0.18
        self.long_shot_reward = 0.5
        self.possession_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward),
                      "possession_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage long-range shots
            is_long_shot_attempt = (
                o['ball_owned_team'] == 0 and
                o['ball'][0] > self.penalty_box_x_threshold and o['ball'][1] <= 0.42 and o['ball'][1] >= -0.42
            )
            if is_long_shot_attempt and o['game_mode'] in [0, 1]:  # Normal play or KickOff
                components["long_shot_reward"][rew_index] = self.long_shot_reward
            
            # Reward possession in the opponent's half
            if o['ball_owned_team'] == 0 and o['ball'][0] > 0:
                components["possession_reward"][rew_index] = self.possession_reward

            reward[rew_index] += components["long_shot_reward"][rew_index] + components["possession_reward"][rew_index]

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
