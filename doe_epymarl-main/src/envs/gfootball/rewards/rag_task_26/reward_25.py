import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a position-based reward."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for central midfield (CM) and wide midfield (WM) role-specific rewards
        self.central_midfield_progress = 0
        self.wide_midfield_crosses = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset all counters upon environment reset."""
        self.central_midfield_progress = 0
        self.wide_midfield_crosses = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the wrapper state for serialization."""
        # Store specific state variables
        to_pickle['central_midfield_progress'] = self.central_midfield_progress
        to_pickle['wide_midfield_crosses'] = self.wide_midfield_crosses
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the wrapper state from de-serialization."""
        from_pickle = self.env.set_state(state)
        self.central_midfield_progress = from_pickle['central_midfield_progress']
        self.wide_midfield_crosses = from_pickle['wide_midfield_crosses']
        return from_pickle

    def reward(self, reward):
        """Reward adjustments for specific midfield player roles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_interaction_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            designated_player_role = o['left_team_roles'][o['designated']]
            active_role = o['left_team_roles'][o['active']]

            if designated_player_role == 5 or active_role == 5:  # Central midfield roles
                # Increment reward for successful transitions
                if o['game_mode'] == 0 and o['ball_owned_team'] == 0:
                    midfield_transitions = abs(o['ball'][0]) < 0.4
                    if midfield_transitions:
                        self.central_midfield_progress += 1
                        components["midfield_interaction_reward"][rew_index] = 0.05

            if designated_player_role in [6, 7] or active_role in [6, 7]:  # Wide midfield roles
                # Increment reward for successful crosses
                if o['game_mode'] == 0 and o['ball_owned_team'] == 0:
                    crosses = abs(o['ball'][1]) > 0.7
                    if crosses:
                        self.wide_midfield_crosses += 1
                        components["midfield_interaction_reward"][rew_index] = 0.1

            reward[rew_index] += components["midfield_interaction_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take a step using the wrapped environment's step method and return processed results."""
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
