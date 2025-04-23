import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages shooting from the central field with accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones for central striking positions on the field.
        self.central_zone_min_x = -0.2
        self.central_zone_max_x = 0.2
        self.central_zone_goal_y = 0.85  # Close to the opponent's goal

        # Reward multipliers
        self.shot_power_reward = 0.2
        

    def reset(self):
        """Reset the wrapper's state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Customize reward based on agent's shooting accuracy and power from central positions."""
        observation = self.env.unwrapped.observation()
        # Initial reward components
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy_reward": [0.0] * len(reward)}
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            is_in_central_zone = (self.central_zone_min_x <= ball_pos[0] <= self.central_zone_max_x)

            if is_in_central_zone and abs(ball_pos[1]) > self.central_zone_goal_y:
                # If the player is in the central zone and close to opponent's goal, check shooting power and direction
                if o['ball_direction'][1] > 0:  # positive y-direction (towards opponent's goal)
                    power = np.linalg.norm(o['ball_direction'])
                    components["shooting_accuracy_reward"][rew_index] = power * self.shot_power_reward
                    reward[rew_index] += components["shooting_accuracy_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Apply actions, process observations, and adjust rewards."""
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
