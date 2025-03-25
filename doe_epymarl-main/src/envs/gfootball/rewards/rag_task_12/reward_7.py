import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards based on midfield and defense-oriented tasks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the state of the environment for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to pickle."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from pickle."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards based on controlled player's behavior towards midfield and defense activities."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),  # Copy the initial rewards
            "high_pass": [0.0] * len(reward),
            "long_pass": [0.0] * len(reward),
            "dribble_under_pressure": [0.0] * len(reward)
        }

        # Ensure we have observations to evaluate
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            active_player_actions = o['sticky_actions']

            # Reward for successful high pass actions
            if active_player_actions[7]:  # Assuming '7' corresponds to a high pass action
                components['high_pass'][i] += 0.1  # Reward increment for high pass

            # Reward for successful long pass actions
            if active_player_actions[9]:  # Assuming '9' corresponds to a long pass action
                components['long_pass'][i] += 0.1

            # Reward for maintaining dribble under pressure
            if active_player_actions[8] and o['game_mode'] in {2, 3}:  # Assuming '8' is dribble under pressure
                components['dribble_under_pressure'][i] += 0.2

            # Combine all rewards components
            reward[i] += (
                components['high_pass'][i] +
                components['long_pass'][i] +
                components['dribble_under_pressure'][i]
            )

        return reward, components

    def step(self, action):
        """Step the environment by the given action."""
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
