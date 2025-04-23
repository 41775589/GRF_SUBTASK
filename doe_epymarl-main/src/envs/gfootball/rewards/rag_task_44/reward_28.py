import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for Stop-Dribble behavior under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To track usage of sticky actions

    def reset(self):
        """
        Resets the environment and the counters for sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Saves the current state of the environment and sticky action counters.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restores the environment state and sticky action counters.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on Stop-Dribble action under pressure situations. 
        It provides a dense reward for switching from dribbling to a stationary position 
        when an opponent is close.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, obs in enumerate(observation):
            distance_to_closest_opponent = np.min(np.linalg.norm(
                obs['right_team'] - obs['left_team'][obs['active']], axis=1))
            has_ball = obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']

            # Check if dribble is on and a sudden stop is made
            if has_ball and obs['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] > 0:
                self.sticky_actions_counter[9] -= 1
                if distance_to_closest_opponent < 0.1:  # Close proximity
                    components["stop_dribble_reward"][rew_index] = 0.3
                    reward[rew_index] += components["stop_dribble_reward"][rew_index]

            # Update sticky actions counting
            self.sticky_actions_counter = obs['sticky_actions']

        return reward, components

    def step(self, action):
        """
        The environment's step function that processes the action, updates observation and reward.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
