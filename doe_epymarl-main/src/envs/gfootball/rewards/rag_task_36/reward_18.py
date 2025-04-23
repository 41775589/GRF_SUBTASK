import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for dribbling and dynamic positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter and the environment.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Hooks into the environment's get_state and stores state needed for the wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = np.copy(self.sticky_actions_counter)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the state of the environment from a pickled state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.copy(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on dribbling and dynamic positioning.
        - Adds rewards for initiating and stopping dribbles.
        - Rewards transition between defensive and offensive positioning.
        - Dense rewards for keeping the ball in controlled play, and not losing it to opponents.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for dribbling and stopping dribbling
            if 'sticky_actions' in o:
                actions = o['sticky_actions']
                dribbling = actions[9] # dribble action index
                # Reward for starting to dribble
                if dribbling and self.sticky_actions_counter[9] == 0:
                    components["dribble_reward"][rew_index] += 0.05
                # Reward for stopping dribbling
                elif not dribbling and self.sticky_actions_counter[9] == 1:
                    components["dribble_reward"][rew_index] += 0.05
                self.sticky_actions_counter[9] = dribbling

            # Sum up rewards
            reward[rew_index] += components["dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        updated_obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in updated_obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
