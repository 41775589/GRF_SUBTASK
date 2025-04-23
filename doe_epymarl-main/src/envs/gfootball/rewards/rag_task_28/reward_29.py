import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense dribbling skill improvement reward, focusing on maintaining proximity and ball control when facing the goalkeeper."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._approach_goalkeeper_reward = 0.3
        self._control_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "approach_goalkeeper_reward": [0.0] * len(reward),
                      "control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate the distance to the opponent’s goalkeeper
            goalkeeper_pos = o.get('right_team')[0]  # Assuming index 0 is always the goalkeeper
            player_pos = o.get('right_team')[o.get('active')]
            goalkeeper_distance = np.linalg.norm(np.array(goalkeeper_pos[:2]) - np.array(player_pos[:2]))

            # Reward for getting closer to goalkeeper
            if goalkeeper_distance < 0.2:  # Threshold for 'closeness'
                components['approach_goalkeeper_reward'][rew_index] = self._approach_goalkeeper_reward
                reward[rew_index] += components['approach_goalkeeper_reward'][rew_index]

            # Reward for maintaining control while dribbling
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == o['active']:  # Assuming index 9 for 'dribble' action
                components['control_reward'][rew_index] = self._control_reward
                reward[rew_index] += components['control_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
