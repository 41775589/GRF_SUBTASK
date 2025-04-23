import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on advanced ball control and passing under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_coeff = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']["sticky_actions_counter"])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Short Pass, High Pass, Long Pass under pressure
            # Check if ball is owned by the active player of left team close to opponent
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and np.any(o['right_team_direction'] != [0, 0]):
                x, y = o['ball']
                
                # Assess pressure based on opponent proximity
                close_opponents = np.sum(np.sqrt(np.sum((o['right_team'] - [x, y])**2, axis=1)) < 0.2)
                
                # Reward passes under pressure
                if close_opponents > 2:  # consider 3 or more opponents as high pressure
                    for action_key, action_value in enumerate(o['sticky_actions'][3:6]):  # indices for pass actions
                        if action_value == 1:
                            components["pass_reward"][rew_index] += self.pass_reward_coeff
                            reward[rew_index] += components["pass_reward"][rew_index]

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
