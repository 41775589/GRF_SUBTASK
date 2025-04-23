import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for optimizing shooting angles and timing under high-pressure scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_power_factor = 0.5
        self.timing_weight = 0.3
        self.angle_weight = 0.7

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "shot_power_reward": [0.0] * len(reward),
                      "timing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o.get('ball_owned_team', -1)
            ball_owned_player = o.get('ball_owned_player', -1)
            if ball_owned_team == 1 and ball_owned_player == o['active']:
                shot_power = min(o['ball_direction'][2], 1)  # Assuming third index is shot power
                timing = 1 - (o['steps_left'] / self.env.unwrapped.spec.max_episode_steps)  # Closer to game end is better timing
                angle = np.abs(o['ball_direction'][1])  # Assuming Y direction gives shot angle

                # Calculate additional rewards
                components["shot_power_reward"][rew_index] = self.shot_power_factor * shot_power
                components["timing_reward"][rew_index] = self.timing_weight * timing
                components["angle_reward"][rew_index] = self.angle_weight * angle

                additional_reward = (components["shot_power_reward"][rew_index] +
                                     components["timing_reward"][rew_index] + 
                                     components["angle_reward"][rew_index])

                reward[rew_index] += additional_reward
        
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
