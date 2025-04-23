import gym
import numpy as np
class ShootingPracticeRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to focus on shooting techniques and timings."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset for new episode and clear sticky_actions_counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return current state with additional environment states added."""
        to_pickle['ShootingPracticeRewardWrapper'] = dict(sticky_actions_counter=self.sticky_actions_counter)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state with additional environment states handled."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['ShootingPracticeRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Custom reward logic focusing on shooting angles and timing under pressure."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shooting_reward": [0.0] * len(reward)}

        # Early exit if there is no observation.
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Check if shooting action is taking place
            if 'action' in o and o['action'] == 'shoot':
                # Calculate the efficiency based on the proximity to the goal and presence of defenders
                goal_proximity = abs(o['ball'][0] - 1)  # Assuming goal at x=1
                defenders_pressure = np.sum((o['right_team'][:, 0] > o['ball'][0]) & (abs(o['right_team'][:, 1] - o['ball'][1]) < 0.1))
                shooting_efficiency = max(0.1, goal_proximity * (1 - 0.1 * defenders_pressure))
                
                components["shooting_reward"][i] = shooting_efficiency
                reward[i] += components["shooting_reward"][i]

        return reward, components

    def step(self, action):
        """Execute an environment step and augment with reward modifications."""
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
