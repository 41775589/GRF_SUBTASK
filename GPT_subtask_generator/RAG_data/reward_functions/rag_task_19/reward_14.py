import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that increases the emphasis on mid-field control and precise defense transitions during game-play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize midfield control and defense transition factors
        self.midfield_control_weight = 0.5
        self.defense_transition_weight = 0.5
        self.reset_metrics()

    def reset(self):
        """Reset all customized reward metrics."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset_metrics()
        return self.env.reset()

    def reset_metrics(self):
        self.midfield_control_score = 0
        self.defensive_transitions_score = 0

    def get_state(self, to_pickle):
        """Get the state from the base class augmented with our custom metrics."""
        super_state = self.env.get_state(to_pickle)
        super_state['midfield_control_score'] = self.midfield_control_score
        super_state['defensive_transitions_score'] = self.defensive_transitions_score
        return super_state

    def set_state(self, state):
        """Set the state using the base class and include our custom metrics."""
        from_pickle = self.env.set_state(state)
        self.midfield_control_score = from_pickle.get('midfield_control_score', 0)
        self.defensive_transitions_score = from_pickle.get('defensive_transitions_score', 0)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on midfield control and defensive transitions."""
        obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "defensive_transitions_reward": [0.0] * len(reward)
        }

        if obs is None:
            return reward, components

        for rew_index, o in enumerate(obs):
            # Reward for controlling midfield
            if np.abs(o['ball'][0]) < 0.3:  # Assuming midfield is central part of the field
                self.midfield_control_score += self.midfield_control_weight
                components["midfield_control_reward"][rew_index] = self.midfield_control_weight

            # Reward for successful defensive transitions
            if o['game_mode'] in [3, 4]:  # Assuming modes 3 and 4 relate to defensive plays like free kicks
                self.defensive_transitions_score += self.defense_transition_weight
                components["defensive_transitions_reward"][rew_index] = self.defense_transition_weight

            # Aggregate all component rewards
            reward[rew_index] += components["midfield_control_reward"][rew_index] + components["defensive_transitions_reward"][rew_index]

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
