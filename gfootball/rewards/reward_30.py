import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for offensive strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize parameters for offensive strategy rewards
        self._successful_pass_reward = 0.1
        self._effective_dribble_reward = 0.2
        self._on_target_shot_reward = 0.3

    def reset(self):
        """Resets the environment and any internal state."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state along with internal wrapper state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state from serialized data, recovering internal state."""
        from_pickle = self.env.set_state(state)
        # Assume some internal state may need to be set from the pickle
        return from_pickle

    def reward(self, reward):
        """Modify reward based on offensive gameplay components."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Evaluating effective pass (assuming some observation indicates effective pass)
            if o.get('effective_pass', False):
                components["pass_reward"][i] = self._successful_pass_reward

            # Evaluating dribbles (assuming some observation indicates successful dribbling)
            if o.get('successful_dribble', False):
                components["dribble_reward"][i] = self._effective_dribble_reward

            # Evaluating shots on target (assuming some observation indicates a shot on target)
            if o.get('on_target_shot', False):
                components["shot_reward"][i] = self._on_target_shot_reward

            # Update the reward for this player
            reward[i] += (components["pass_reward"][i] 
                          + components["dribble_reward"][i] 
                          + components["shot_reward"][i])

        return reward, components

    def step(self, action):
        """Performs a game environment step and augments observables with reward analysis."""
        observation, reward, done, info = self.env.step(action)
        
        # Modify the rewards after the step based on custom strategy
        reward, components = self.reward(reward)
        
        # Add detailed reward components to info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
