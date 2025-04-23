import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific midfield dynamics rewards based on ball control and midfield player contributions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control = 0
        self.midfield_checkpoint_reward = 0.1
        self.max_control_count = 10  # max counts to control the midfield region

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['MidfieldControl'] = self.midfield_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control = from_pickle['MidfieldControl']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Midfield control based on the width of the midfield area and controlling the ball
            if ('ball_owned_team' in o and o['ball_owned_team'] == o['active']) and \
               (o['right_team'][o['active']][0] > -0.25 and o['right_team'][o['active']][0] < 0.25):
                self.midfield_control += 1
                if self.midfield_control <= self.max_control_count:
                    components["midfield_reward"][rew_index] = self.midfield_checkpoint_reward
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
