import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling maneuvers and dynamic positioning."""

    def __init__(self, env):
        super().__init__(env)
        # Details of the dribble states of each agent
        self.dribble_state = np.zeros(2, dtype=bool)  # Two players, True if dribbling
        self.positional_reward_scale = 0.1  # Scale for positional rewards
        self.dribble_reward_scale = 0.05  # Scale for dribble rewards
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking sticky actions

    def reset(self):
        self.dribble_state.fill(False)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'dribble_state': self.dribble_state.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle.get('CheckpointRewardWrapper', {})
        self.dribble_state = np.array(saved_state.get('dribble_state', [False, False]))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0, 0.0],
                      "dribble_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]

            # Positional reward based on y-position (center is better)
            y_pos = abs(o['right_team'][o['active']][1])  # Considering vertical position
            components["positional_reward"][idx] = (1 - y_pos) * self.positional_reward_scale
            reward[idx] += components["positional_reward"][idx]

            # Dribbling rewards
            dribbling_action = o['sticky_actions'][9]  # 'action_dribble' is at index 9
            if dribbling_action == 1 and not self.dribble_state[idx]:
                self.dribble_state[idx] = True
                reward[idx] += 0.1  # Reward starting a dribble
            if dribbling_action == 0 and self.dribble_state[idx]:
                self.dribble_state[idx] = False
                reward[idx] += 0.05  # Reward stopping a dribble correctly

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
