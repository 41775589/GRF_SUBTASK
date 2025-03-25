import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defense-focused checkpoint reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = np.zeros(2, dtype=int)  # Track defensive positions per agent
        self.defensive_rewards = np.array([0, 0], dtype=float)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions.fill(0)
        self.defensive_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions.tolist()
        to_pickle['defensive_rewards'] = self.defensive_rewards.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions = np.array(from_pickle['defensive_positions'])
        self.defensive_rewards = np.array(from_pickle['defensive_rewards'])
        return from_pickle

    def reward(self, reward):
        # This rewarding method enhances defensive behavior
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": self.defensive_rewards.copy()
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Detect actions like Sliding, Stop-Dribble, and Stop-Moving
            defensive_actions = o['sticky_actions'][7:10]  # indexes definitions should match action meanings

            # Calculate defense position index, region is based on proximity to own goal
            x_position = o['left_team'][o['active']][0]  # X value of the active player in left team
            defense_position_index = min(int((x_position + 1) * 5), 9)  # Normalize X to range [0, 10)

            if np.any(defensive_actions):
                # Reward if new defensive position is better and a defensive action is being performed
                if defense_position_index > self.defensive_positions[rew_index]:
                    increment = (defense_position_index - self.defensive_positions[rew_index]) * 0.05
                    components['defensive_reward'][rew_index] += increment
                    reward[rew_index] += increment
                    self.defensive_positions[rew_index] = defense_position_index
            
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
