import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper focusing on enhancing the midfield control and play-making contributions.
    Rewards agents based on their effectiveness in controlling the midfield area.
    """

    def __init__(self, env):
        super().__init__(env)
        self.midfield_control_coefficient = 0.05
        self.midfield_activation_positions = [-0.25, 0.25]
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x, ball_y = o['ball'][:2]

            # Control midfield
            if o['active'] >= 0:
                player_x = o[f'{"left" if rew_index % 2 == 0 else "right"}_team'][o['active']]
                if player_x[0] > self.midfield_activation_positions[0] and player_x[0] < self.midfield_activation_positions[1]:
                    components["midfield_reward"][rew_index] = self.midfield_control_coefficient * abs(ball_x - player_x[0])

            reward[rew_index] += components["midfield_reward"][rew_index]

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
