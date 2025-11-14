import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward mechanism focusing on quick counter-attacks via strategic long passes and sprint transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_action = None

    def reset(self):
        """Reset the environment and clearing action trackers."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_action = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store custom wrapper state in the pickle object."""
        state = self.env.get_state(to_pickle)
        state['previous_action'] = self.previous_action
        return state

    def set_state(self, state):
        """Retrieve and set custom wrapper state from the pickle object."""
        self.previous_action = state.get('previous_action', None)
        return self.env.set_state(state)

    def reward(self, reward):
        """Calculate and assign rewards based on completed actions relevant to counter-attacking strategy."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0],
            "sprint_transition_reward": [0.0]
        }

        obs = observation[0]
        game_mode = obs['game_mode']
        current_action = obs['active']

        # Reward for executing long passes
        if game_mode in [0, 2, 4, 5] and current_action == 9:  # Assuming index 9 is Long Pass
            components["long_pass_reward"][0] = 1.0

        # Reward for sprint transitions
        if self.previous_action is not None and self.previous_action == 5 and current_action == 8:  # from Stop to Sprint
            components["sprint_transition_reward"][0] = 1.5

        # Calculate total reward
        reward[0] += components["long_pass_reward"][0] + components["sprint_transition_reward"][0]

        # Update previous action for the next step
        self.previous_action = current_action

        return reward, components

    def step(self, action):
        """Process actions, rewards and returns results with modified reward as per wrapping."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
