import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing transitional play and pace management by central midfielders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.episode_transitions = 0
        self.transition_threshold = 0.1  # Threshold for what counts as significant ball movement
        self.midfield_control_coeff = 1.5  # Reward multiplier for effective midfield control
        self.pace_management_coeff = 0.5  # Reward modifier for maintaining optimal pace

    def reset(self):
        self.episode_transitions = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['transitions_in_game'] = self.episode_transitions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.episode_transitions = from_pickle['transitions_in_game']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "pace_management_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Assess ball movement and field positioning
            midfield_area = (-0.2 <= o['ball'][0] <= 0.2)  # Simplified midfield area
            if midfield_area:
                # Increasing rewards for effective ball passing in the midfield zone
                components["midfield_control_reward"][idx] = self.midfield_control_coeff
                reward[idx] += components["midfield_control_reward"][idx]

            # Calculate the ball movement magnitude to manage pace
            magnitude = np.linalg.norm(o['ball_direction'][:2])
            if magnitude < self.transition_threshold:
                # Reward for maintaining composure and controlled ball movements (pace management)
                components["pace_management_reward"][idx] = self.pace_management_coeff * (self.transition_threshold - magnitude)
                reward[idx] += components["pace_management_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Update the info dict with the components of the reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
