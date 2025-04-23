import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ Custom reward wrapper for Stop-Sprint and Stop-Moving techniques. """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset sticky actions count on environment reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Retrieve state from the environment and add specific wrapper info
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the state of the environment and also restore wrapper-specific info
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """ Enhance reward function focusing on precise stop-sprint and stop-moving abilities. """
        # Retrieve observations from environment
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "stop_move_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Initialize reward component
            stop_reward = 0.0

            # Check if the player had stopped or performing a sprint action:
            if o['sticky_actions'][8] == 1:  # Sprint is active
                stop_reward += 0.2
                self.sticky_actions_counter[8] += 1
            if o['sticky_actions'][9] == 1:  # Dribble is active
                stop_reward += 0.1
                self.sticky_actions_counter[9] += 1
            
            # Award stop-start behavior involving sudden stops and sprints
            if self.sticky_actions_counter[8] > 1 and self.sticky_actions_counter[9] > 1:
                reward[rew_index] += stop_reward

            components["stop_move_reward"][rew_index] = stop_reward

        return reward, components

    def step(self, action):
        ''' Performs the environment's step, evaluating reward adjustment from this wrapper. '''
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        # For debugging purposes, record all sticky actions
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
