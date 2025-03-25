import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies."""

    def __init__(self, env):
        """
        Initialize the reward wrapper.
        :param env: the environment to wrap.
        """
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for practicing different pass types and dribbling
        self.pass_reward = 0.2
        self.dribble_reward = 0.3
        self.shot_reward = 0.5

    def reset(self):
        """Reset the environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of the environment with wrapper-specific data."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from saved data."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Reward function that promotes offensive strategies.
        :param reward: list of base rewards from the environment.
        """
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observations[rew_index]
            
            # If a shot is made
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming 9 is the index for the shoot action
                components["shot_reward"][rew_index] = self.shot_reward
                reward[rew_index] += components["shot_reward"][rew_index]

            # Check dribbling action
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming dribble is at index 9
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Reward for passing
            if o.get('right_team_direction', None) is not None:
                # Vector distance could imply a pass
                distances = [np.linalg.norm(o['right_team_direction'][i] - o['ball_direction'])
                             for i in range(len(o['right_team_direction']))]
                if any(d < 0.5 for d in distances):  # Threshold for considering a successful pass
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take an environment step and modify the reward.
        :param action: the actions to be taken.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
