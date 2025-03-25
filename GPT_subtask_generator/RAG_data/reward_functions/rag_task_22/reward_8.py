import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for sprinting efficiently across the field."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sprint_rewards = {}
        self._sprint_threshold = 0.5

    def reset(self):
        """Reset the environment state and reward data."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sprint_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Safely extract state for serialization."""
        to_pickle['SprintRewardWrapper'] = self._sprint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from deserialized state."""
        from_pickle = self.env.set_state(state)
        self._sprint_rewards = from_pickle['SprintRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute reward by adding sprinting efficiency bonuses."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' in o:
                sprint_action_active = o['sticky_actions'][8] == 1  # Action index 8 corresponds to sprinting
                if sprint_action_active:
                    if rew_index not in self._sprint_rewards:
                        self._sprint_rewards[rew_index] = 0
                    # Increment sprint count only if within the sprint threshold
                    active_player_position = np.array(o['right_team'] if o['ball_owned_team'] == 1 else o['left_team'])[o['active']]
                    ball_position = np.array(o['ball'][:2])
                    dist = np.linalg.norm(active_player_position - ball_position)

                    # Reward for being near the ball and sprinting effectively
                    if dist < self._sprint_threshold:
                        additional_reward = 0.1
                        self._sprint_rewards[rew_index] += additional_reward
                        components["sprint_reward"][rew_index] += additional_reward

            reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment while tracking rewards computation."""
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
