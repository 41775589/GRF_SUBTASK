import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for transitioning from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_count = 0
        self.dribble_count = 0
        self.prev_ball_owned_team = -1

    def reset(self):
        """Reset the environment and reward counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_count = 0
        self.dribble_count = 0
        self.prev_ball_owned_team = -1
        return self.env.reset()

    def reward(self, reward):
        """Calculate the custom reward based on ball control and transitions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        reward_components = {
            "base_score_reward": reward.copy(),
            "pass_reward": 0.0,
            "dribble_reward": 0.0,
        }
        
        if observation['ball_owned_team'] == 1 and self.prev_ball_owned_team == 0:
            # Transition from defense (team 0) to attack (team 1)
            reward_components['pass_reward'] = 0.2 * self.pass_count
            reward_components['dribble_reward'] = 0.3 * self.dribble_count

        reward += reward_components['pass_reward'] + reward_components['dribble_reward']

        # Update counters and state
        self.prev_ball_owned_team = observation['ball_owned_team']
        self.pass_count = 0
        self.dribble_count = 0

        sticky_actions = observation['sticky_actions']
        if sticky_actions[8]:  # action_dribble
            self.dribble_count += 1
        if sticky_actions[6] or sticky_actions[9]:  # action_long_pass or action_short_pass
            self.pass_count += 1

        return reward, reward_components

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def step(self, action):
        """Take a step using the provided actions, calculate reward, and return observation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
