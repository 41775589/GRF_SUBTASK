import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive maneuvers, 
    specifically focusing on sliding tackles under high-pressure situations."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_for_tackling = 0.5
        # Tracking tackles
        self.tackles_attempted = 0
        self.successful_tackles = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_attempted = 0
        self.successful_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['tackles_attempted'] = self.tackles_attempted
        to_pickle['successful_tackles'] = self.successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        self.tackles_attempted = from_pickle.get('tackles_attempted', self.tackles_attempted)
        self.successful_tackles = from_pickle.get('successful_tackles', self.successful_tackles)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' in o:
                # Detect sliding tackle action (assuming index 7 might be sliding tackle)
                if o['sticky_actions'][7] == 1:
                    self.tackles_attempted += 1
                    # Adding successful tackle detection logic
                    if o['ball_owned_team'] != o['active']:
                        # The agent successfully tackled and gained the ball
                        components["tackle_reward"][rew_index] = self.reward_for_tackling
                        reward[rew_index] += components["tackle_reward"][rew_index]
                        self.successful_tackles += 1

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
