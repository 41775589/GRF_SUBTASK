import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for mastering wide midfield roles, focusing on High Pass and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking of sticky actions.
        self.pass_quality_reward = 0.05  # Incremental reward for successful high passes.
        self.positioning_reward = 0.1    # Reward for good positioning in wide areas.

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current internal state."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on wide midfield roles."""
        observation = self.env.unwrapped.observation()  # Get the raw observations from the environment
        components = {
            "base_score_reward": np.array(reward, dtype=float),
            "pass_quality_reward": np.zeros(len(reward)),
            "positioning_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components
        
        # Loop through observations and calculate rewards
        for i, obs in enumerate(observation):
            if 'sticky_actions' in obs:
                high_pass_action = obs['sticky_actions'][9]  # Assuming index 9 corresponds to 'action_high_pass'
                if high_pass_action == 1:
                    components["pass_quality_reward"][i] += self.pass_quality_reward
                    reward[i] += components["pass_quality_reward"][i]

            if 'right_team_roles' in obs:
                # Look for wide midfield position (7 = Right Midfield, 6 = Left Midfield)
                if obs['right_team_roles'][obs['active']] in [6, 7]:
                    # Evaluate the positioning in terms of y-coordinate being near sidelines
                    if abs(obs['right_team'][obs['active'], 1]) > 0.3:  # Assuming field width from -0.42 to 0.42
                        components["positioning_reward"][i] += self.positioning_reward
                        reward[i] += components["positioning_reward"][i]

            self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions at each step

        return reward, components

    def step(self, action):
        """Step method to apply the custom reward and add it in the info dictionary."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
