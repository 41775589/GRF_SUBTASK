import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on the task of a 'sweeper' role in football, adept at clearing the ball from the
    defensive zone, performing critical last-man tackles, and supporting the stopper by covering positions and
    executing fast recoveries. This wrapper encourages defensive actions and ball clearances.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 1.0
        self.tackle_reward = 0.5
        self.position_support_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, from_picle):
        state = self.env.set_state(from_picle)
        self.sticky_actions_counter = from_picle['sticky_actions_counter']
        return state

    def reward(self, reward):
        """
        Enhance the reward based on defensive actions:
        - Increase reward for ball clearances.
        - Reward for tackles near the goal to prevent opposing scores.
        - Position support reward for covering positions that are critical in defensive strategies.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'clearance_reward': [0.0] * len(reward),
                      'tackle_reward': [0.0] * len(reward),
                      'position_support_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            # Reward clearances
            if o['game_mode'] == 2:  # Assuming game_mode 2 indicates a clearance situation
                components['clearance_reward'][idx] = self.clearance_reward
                reward[idx] += self.clearance_reward

            # Reward tackles
            if 'tackle' in o['sticky_actions'] and o['sticky_actions']['tackle']:
                components['tackle_reward'][idx] = self.tackle_reward
                reward[idx] += self.tackle_reward

            # Reward positional support
            if not o['ball_owned_team'] and self.is_sweeping_position(o):
                components['position_support_reward'][idx] = self.position_support_reward
                reward[idx] += self.position_support_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def is_sweeping_position(self, observation):
        # Assuming there's a metric or a condition that defines the sweeper's effective position
        # This is a placeholder for actual logic that should check if the player is in a typical 'sweeper' position.
        return True
