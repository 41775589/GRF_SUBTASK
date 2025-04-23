import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward related to defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize rewards for defensive intercepts and positioning
        self.intercept_reward = 0.5
        self.position_reward = 0.3

    def reset(self):
        """Reset the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get state for saving and restoring the environment."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from a restored state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Modifies reward based on defensive actions demonstrated by agents."""
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 3:  # FreeKick/Defensive situation
                if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                    components["intercept_reward"][rew_index] = self.intercept_reward
                    reward[rew_index] += components["intercept_reward"][rew_index]

            # Positional awareness - closer to own goal when opponent possesses the ball
            if o['ball_owned_team'] ==1:
                distance_from_goal = np.linalg.norm(o['right_team'][o['active']] + np.array([1, 0]))
                components["position_reward"][rew_index] = self.position_reward / max(distance_from_goal, 0.1)
                reward[rew_index] += components["position_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Environment step with modified rewards including defensive components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
