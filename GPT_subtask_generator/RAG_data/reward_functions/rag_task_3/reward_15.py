import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on shooting with accuracy and power under pressure."""
    
    def __init__(self, env, precision_threshold=0.1, power_threshold=0.5):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions (like sprint & dribble)
        self.precision_threshold = precision_threshold  # Proximity to the goal to consider as precise
        self.power_threshold = power_threshold  # Ball speed above this value considered as powerful shot
        self._goal_reward = 5.0  # Reward for scoring a goal
        self._precision_reward = 2.0  # Reward for shooting near the goal

    def reset(self):
        """Reset the wrapper and the environment."""
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state while including the wrapper information."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state while including the wrapper information."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on shooting criteria (precision and power)."""
        observation = self.env.unwrapped.observation()  # Assumes observation method exposes internals
        base_score_reward = reward.copy()
        precision_reward = [0.0] * len(reward)
        power_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'precision_reward': precision_reward, 'power_reward': power_reward}

        for rew_index, _ in enumerate(reward):
            o = observation[rew_index]
            ball_speed = np.linalg.norm(o['ball_direction'])
            
            # Check if score has occurred
            if o['score'][1] > o['score'][0]:  # Assuming scored against the right side is the main objective
                precision_reward[rew_index] = self._goal_reward
                reward[rew_index] += precision_reward[rew_index]
            
            # Additional reward if the shot is both precise and powerful
            if o['game_mode'] == 2:  # Considering game_mode 2 as a shot action
                goal_distance = np.linalg.norm([o['ball'][0] - 1, o['ball'][1]])  # Distance to right goal based on ball's x, y
                if goal_distance <= self.precision_threshold:
                    precision_reward[rew_index] = self._precision_reward
                    reward[rew_index] += precision_reward[rew_index]
                if ball_speed > self.power_threshold:
                    power_reward[rew_index] = self._precision_reward
                    reward[rew_index] += power_reward[rew_index]

        components = {'base_score_reward': base_score_reward, 'precision_reward': precision_reward, 'power_reward': power_reward}
        return reward, components

    def step(self, action):
        """Step through the environment and modify output rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
