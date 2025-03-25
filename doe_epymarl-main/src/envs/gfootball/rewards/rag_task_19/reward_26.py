import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for defensive and midfield play control."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}
        self.midfield_rewards = {}
        self.num_defensive_zones = 5
        self.num_midfield_zones = 5
        self.defensive_reward_value = 0.2
        self.midfield_reward_value = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}
        self.midfield_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveRewards'] = self.defensive_rewards
        to_pickle['MidfieldRewards'] = self.midfield_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle['DefensiveRewards']
        self.midfield_rewards = from_pickle['MidfieldRewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_rewards": [0.0] * len(reward),
                      "midfield_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            ball_position = o['ball']
            player_position = o['left_team'][o['active']] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])]

            # Calculate distance from defensive and midfield zones
            dist_to_def_zone = np.abs(ball_position[0] + 0.7)  # Simulated defensive zone boundary at x = 0.7
            dist_to_midfield_zone = np.abs(ball_position[0])    # Midfield zone around x = 0

            # Assign defensive zone rewards
            if dist_to_def_zone < 0.1 and rew_index not in self.defensive_rewards:
                components['defensive_rewards'][rew_index] = self.defensive_reward_value
                self.defensive_rewards[rew_index] = 1

            # Assign midfield zone rewards
            if dist_to_midfield_zone < 0.1 and rew_index not in self.midfield_rewards:
                components['midfield_rewards'][rew_index] = self.midfield_reward_value
                self.midfield_rewards[rew_index] = 1

            # Update the base reward with additional components from defensive and midfield play
            reward[rew_index] += (components['defensive_rewards'][rew_index] +
                                  components['midfield_rewards'][rew_index])

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
