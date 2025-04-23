import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces a reward for stopping or starting movements quickly 
    to facilitate rapid defensive adaptations to offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Parameters for the defensive training task
        self._num_checkpoints = 5
        self._transition_reward = 0.1  # Reward for effective stop/start transition
        self._previous_action = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_action = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = dict(previous_action=self._previous_action)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._previous_action = from_pickle['CheckpointRewardWrapper']['previous_action']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation), "Mismatch in reward and observation length."

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Extract current and previous actions
            current_action = o['sticky_actions']
            components["transition_reward"][rew_index] = 0
            
            if self._previous_action is not None:
                # Calculate the transition dynamic
                transitions = np.abs(current_action - self._previous_action[rew_index])
                
                # Count rapid stop/start transitions
                rapid_transitions = np.sum(transitions)
                if rapid_transitions > 0:
                    components["transition_reward"][rew_index] = self._transition_reward * rapid_transitions
                    reward[rew_index] += components["transition_reward"][rew_index]

            self._previous_action = current_action
        
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
