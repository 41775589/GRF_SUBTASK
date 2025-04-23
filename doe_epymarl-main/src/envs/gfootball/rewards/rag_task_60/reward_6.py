import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive adaptation reward focusing on transitions between movement states."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 10
        self._transition_reward = 0.05

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores the state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'], dtype=int)
        return from_pickle

    def reward(self, reward):
        """Modifies the rewards based on defensive transitions."""
        
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            previous_action_state = self.sticky_actions_counter != 0
            current_action_state = o['sticky_actions'] != 0

            # Reward for starting or stopping movement
            transitions = np.logical_xor(previous_action_state, current_action_state)
            num_transitions = np.sum(transitions)

            # Calculate transition rewards
            components["transition_reward"][rew_index] = num_transitions * self._transition_reward
            reward[rew_index] += components["transition_reward"][rew_index]

            # Save the current sticky actions as the previous state for the next step
            self.sticky_actions_counter = o['sticky_actions'].copy()

        return reward, components

    def step(self, action):
        """Processes the step actions and calculates reward and components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
