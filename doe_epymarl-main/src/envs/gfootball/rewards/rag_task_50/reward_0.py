import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for making successful long passes between different playfield zones."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Array to keep track of the use of sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_quality_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the player is in control of the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Define regions by x-coordinates (aspect ratio normalized to [-1, 1])
                own_zone = -0.33
                mid_zone = 0.33
                player_x_pos = o['left_team'][o['active']][0]

                # Detect if a long pass has been made
                if player_x_pos < own_zone:  # In defensive third
                    # Assume pass if ball_direction x-component is positive and substantial
                    if o['ball_direction'][0] > 0.2:
                        boundary_passed = any(p[0] > mid_zone for p in o['left_team'] if np.linalg.norm(p - o['ball'][:2]) > 0.3)
                        if boundary_passed:
                            components['passing_quality_reward'][rew_index] += 0.5  # Reward for successful passage from defense to attack third
                elif player_x_pos > mid_zone:  # In attacking third
                    if o['ball_direction'][0] < -0.2:
                        boundary_passed = any(p[0] < own_zone for p in o['left_team'] if np.linalg.norm(p - o['ball'][:2]) > 0.3)
                        if boundary_passed:
                            components['passing_quality_reward'][rew_index] += 0.5  # Reward for successful passage from attack to defense third

            # Combine component rewards with the base score reward
            reward[rew_index] += components['passing_quality_reward'][rew_index] * 1.0  # Coefficient for modifying importance of passing reward

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
                if action == 1:
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
