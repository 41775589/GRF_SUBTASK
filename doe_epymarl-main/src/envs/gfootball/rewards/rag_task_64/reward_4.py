import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes and crosses effectively."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to adjust reward scaling
        self.high_pass_reward = 0.4   # Reward for executing a high pass
        self.crossing_reward = 0.5    # Reward for successful crosses towards the box
        self.pass_target_accuracy = 0.1  # Spatial accuracy needed for successful pass

    def reset(self):
        """Reset the environment and reset the sticky actions counter."""
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Adjust reward based on successful high passes and accurate crosses
        received from actions effective in dynamic attacking plays.
        """
        # Fetch the current observations from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward), "crossing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Expect the observation to hold two observations
        assert len(reward) == len(observation)

        # Iterate over both observations
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if high pass action was executed by the active player
            if 'action' in o and o['action'] == 'high_pass' and o['ball_owned_team'] == 1:
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Evaluate if this action was a successful cross
            if 'action' in o and o['action'] == 'cross':
                ball_end_pos = np.array([o['ball'][0], o['ball'][1]])
                box_zone_target = np.array([1, 0.2])  # Assumed opponent box zone center position
                distance_to_target = np.linalg.norm(ball_end_pos - box_zone_target)

                if distance_to_target <= self.pass_target_accuracy:
                    components["crossing_reward"][rew_index] = self.crossing_reward
                    reward[rew_index] += components["crossing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Apply the action, modify the reward, and return observations."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
