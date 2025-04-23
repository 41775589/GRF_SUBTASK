import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing team synergy through rewards 
    for strategic possession changes and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_transition_reward = 0.1
        self.negative_transition_reward = -0.05
        self.previous_possession = None

    def reset(self):
        """Reset the sticky actions counter and possession status."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_possession'] = self.previous_possession
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from loaded state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        self.previous_possession = from_pickle.get('previous_possession', self.previous_possession)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on strategic positioning and timely possession changes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        current_possession = observation[0]['ball_owned_team']
        if self.previous_possession is not None and self.previous_possession != current_possession:
            for rew_index in range(len(reward)):
                if current_possession == 1 - observation[rew_index]['ball_owned_team']:
                    reward[rew_index] += self.negative_transition_reward
                    components["possession_change_reward"][rew_index] = self.negative_transition_reward
                else:
                    reward[rew_index] += self.positive_transition_reward
                    components["possession_change_reward"][rew_index] = self.positive_transition_reward
        
        self.previous_possession = current_possession
        return reward, components

    def step(self, action):
        """Step through environment with modified reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
