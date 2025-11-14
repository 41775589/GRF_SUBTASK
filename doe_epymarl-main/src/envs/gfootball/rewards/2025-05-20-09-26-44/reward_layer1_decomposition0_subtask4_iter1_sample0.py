import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Custom reward wrapper focusing on defensive and quick transition subtasks with improved metrics."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.5  # Increased reward for successful defense
        self.transition_reward = 0.7  # Increased reward for quick transitions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_checkpoint_reward_state'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['defensive_checkpoint_reward_state']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_actions_reward": [0.0] * len(reward),
                      "transition_actions_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward interception of the ball
            if o['ball_owned_team'] == 1 and not np.any(o['sticky_actions'][[0, 4]]):
                components["defensive_actions_reward"][rew_index] += self.interception_reward
                reward[rew_index] += components["defensive_actions_reward"][rew_index]

            # Reward quick transition through sprint or dribble if ball is being moved forward
            if np.any(o['sticky_actions'][8:10]) and o['ball_direction'][0] > 0.02:
                components["transition_actions_reward"][rew_index] += self.transition_reward
                reward[rew_index] += components["transition_actions_reward"][rew_index]

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
