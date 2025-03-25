import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to encourage the precise timing of sliding tackles under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_successful = False
        self.tackle_attempts = 0

    def reset(self):
        """Reset the internal status at the beginning of an episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_successful = False
        self.tackle_attempts = 0
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward based on the effectiveness and timing of sliding tackles."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'tackle_timing_reward': [0.0] * len(reward)}

        if observation is None or 'ball_owned_team' not in observation:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            # Sliding tackle observed if 'action_bottom_right' is activated
            if 'sticky_actions' in obs and obs['sticky_actions'][9] == 1:
                self.tackle_attempts += 1
                # Check if ball is nearby and in possession of opposite team
                if obs['ball_owned_team'] == 1 - obs.get('active_team', 0):
                    ball_distance = np.linalg.norm(np.array(obs['ball'][:2]) - np.array(obs['left_team'][obs['active']]))
                    if ball_distance < 0.1:  # choosing a threshold for "close to ball"
                        reward[i] += 1.0  # Positive reward for good timing
                        self.tackle_successful = True
                        components['tackle_timing_reward'][i] = 1.0

        return reward, components

    def step(self, action):
        """Step function processing the action and updating rewards."""
        observation, reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(reward)

        info["final_reward"] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions usage
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for j, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{j}"] = action

        return observation, modified_reward, done, info

    def get_state(self, to_pickle):
        """Include custom state in the pickle."""
        to_pickle['tackle_successful'] = self.tackle_successful
        to_pickle['tackle_attempts'] = self.tackle_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore custom state from the pickle."""
        from_pickle = self.env.set_state(state)
        self.tackle_successful = from_pickle['tackle_successful']
        self.tackle_attempts = from_pickle['tackle_attempts']
        return from_pickle
