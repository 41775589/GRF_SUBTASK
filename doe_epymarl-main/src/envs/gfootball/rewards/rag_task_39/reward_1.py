import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for clearing the ball under pressure in defensive zones."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.5
        self.pressure_threshold = 0.2  # hypothetically, distance threshold for "under pressure"

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Base reward provided by the environment
        components = {"base_score_reward": reward.copy()}
        
        # Observe the current state of the environment
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward

        components["clearance_reward"] = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is in a defensive zone and under pressure
            if o['ball'][0] < -0.5:  # Assume left team is the user's team
                # Get distance to nearest opponent
                min_dist = np.min(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1))

                # Check the pressure condition and if the ball was cleared
                if min_dist <= self.pressure_threshold:
                    if any(action == 'clear_ball' for action in o['sticky_actions']):
                        components["clearance_reward"][rew_index] = self.clearance_reward
                        reward[rew_index] += components["clearance_reward"][rew_index]

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
