import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages defending strategies including tackling, efficient movement, and pressured passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward multipliers or constants for specific behaviors
        self.tackling_reward = 0.2
        self.movement_efficiency_reward = 0.1
        self.pressured_passing_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackling_reward": [0.0] * len(reward),
                      "movement_efficiency_reward": [0.0] * len(reward),
                      "pressured_passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for tackling: incentivize regaining ball possession
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["tackling_reward"][rew_index] = self.tackling_reward

            # Reward for movement control: player is not moving (low velocity magnitude)
            if np.linalg.norm(o['left_team_direction'][o['active']]) < 0.01:
                components["movement_efficiency_reward"][rew_index] = self.movement_efficiency_reward

            # Reward for pressured passing: making a pass under pressure
            if o['game_mode'] in {3, 5} and o['ball_owned_team'] == 0:  # Free Kick or Throw In
                components["pressured_passing_reward"][rew_index] = self.pressured_passing_reward

            # Update reward with additional components
            reward[rew_index] += (components["tackling_reward"][rew_index] +
                                  components["movement_efficiency_reward"][rew_index] +
                                  components["pressured_passing_reward"][rew_index])

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
