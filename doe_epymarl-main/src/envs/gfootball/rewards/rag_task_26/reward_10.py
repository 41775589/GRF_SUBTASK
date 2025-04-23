import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces a midfield mastery reward emphasizing transitional control and player roles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize checkpoints related to midfield contributions
        self.midfield_checkpoints = {}
    
    def reset(self):
        """Resets the environment and midfield checkpoints."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state from the environment and adds midfield checkpoints."""
        to_pickle['midfield_checkpoints'] = self.midfield_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the environment and updates internal checkpoints."""
        from_pickle = self.env.set_state(state)
        self.midfield_checkpoints = from_pickle['midfield_checkpoints']
        return from_pickle

    def reward(self, reward):
        """Modifies the rewards based on midfield dynamics mastery."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_play_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            midfield_control_factor = self._assess_midfield_control(obs)
            midfield_transitional_factor = self._transition_quality(obs)

            # Calculating the midfield play contribution to the reward
            midfield_contribution = 0.05 * midfield_control_factor + 0.05 * midfield_transitional_factor
            components["midfield_play_reward"][index] = midfield_contribution
            reward[index] += midfield_contribution

        return reward, components

    def step(self, action):
        """Takes a step in the environment and adjusts reward based on defined reward components."""
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

    def _assess_midfield_control(self, obs):
        """Assesses midfield control based on player positions and ball possession."""
        control_score = 0
        midfield_zone = [-0.3, 0.3]  # Define midfield zone on x-axis
        if 'left_team' in obs and 'ball_owned_team' in obs:
            team = 'left_team' if obs['ball_owned_team'] == 0 else 'right_team'
            for player in obs[team]:
                if midfield_zone[0] <= player[0] <= midfield_zone[1]:
                    control_score += 1
        return control_score

    def _transition_quality(self, obs):
        """Estimates quality of transitions based on ball movements and player roles near midfield."""
        transition_score = 0
        if 'ball_direction' in obs:
            x, y = obs['ball_direction'][0], obs['ball_direction'][1]
            # Reward forward and wide transitions
            if abs(x) > 0.1 and abs(y): # Simplified model for transition quality
                transition_score = 1
        return transition_score
