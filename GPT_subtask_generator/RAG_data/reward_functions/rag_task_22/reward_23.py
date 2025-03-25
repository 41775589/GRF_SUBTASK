import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for fast repositioning using sprinting to improve defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # counters for sticky actions
        self._num_zones = 5  # number of zones for sprinting reward
        self.zone_rewards = np.linspace(0.1, 0.5, self._num_zones)  # linearly increasing reward values for zones

    def reset(self):
        """Reset the sticky action counters and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save checkpoint rewards alongside the environment state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore checkpoint rewards alongside the environment state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Modify the reward function to include sprint utilization for better defensive positioning."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            sprint_action = player_obs['sticky_actions'][8]  # index 8 corresponds to sprint action
            ball_position = player_obs['ball'][0]  # x position of the ball
            player_position = player_obs['left_team'][player_obs['active']][0]  # x position of active player

            # Increase sprint reward based on proximity to the ball when sprinting is active
            if sprint_action:
                self.sticky_actions_counter[8] += 1
                distance_to_ball = np.abs(ball_position - player_position)
                zone_index = min(int(distance_to_ball / 0.2), self._num_zones - 1)  # limiting the max zone index
                components['sprint_reward'][rew_index] += self.zone_rewards[zone_index]

            reward[rew_index] += components['sprint_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        """Step the environment and modify the reward with the new reward components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)  # reset sticky actions after every step
        return observation, reward, done, info
