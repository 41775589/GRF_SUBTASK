import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting accuracy and power during game simulations."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_power_threshold = 0.9  # Adjusts the threshold for considering high power shots
        self.shot_accuracy_reward = 0.2   # Rewards for being in a position to shoot accurately
        self.shot_power_reward = 0.2      # Rewards for hitting the shot with high power

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        components = {
            "base_score_reward": reward.copy(),
            "shot_accuracy_reward": [0.0] * len(reward),
            "shot_power_reward": [0.0] * len(reward)
        }
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for high power shots
            power = np.linalg.norm(o['ball_direction'][:2])  # Assuming 2D velocity in ball_direction
            if power > self.shot_power_threshold:
                components['shot_power_reward'][rew_index] = self.shot_power_reward
            
            # Reward for shooting towards the goal accurately
            ball_position = o['ball'][0]  # X Position of the ball
            if o['ball_owned_team'] == 0 and ball_position > 0.8:  # Simplified accuracy condition on field positioning
                components['shot_accuracy_reward'][rew_index] += self.shot_accuracy_reward
            
            # Apply the calculated component rewards to the total reward
            reward[rew_index] += (components['shot_accuracy_reward'][rew_index] +
                                  components['shot_power_reward'][rew_index])

        return reward, components

    def step(self, action):
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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
