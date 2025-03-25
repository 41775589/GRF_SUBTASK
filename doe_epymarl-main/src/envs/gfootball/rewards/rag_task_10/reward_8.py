import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance training on defensive skills such as positioning, interception, marking, and tackling."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.defensive_achievement_counter = np.zeros((2, 5), dtype=int)  # Consider a 5x2 matrix for 2 players per 5 defensive tasks
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = np.array([0.2, 0.1, 0.15, 0.25, 0.3])  # Reward values for different defensive actions

    def reset(self):
        self.defensive_achievement_counter = np.zeros((2, 5), dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_achievement_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_achievement_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for agent_idx, agent_obs in enumerate(observation):
            defensive_score = 0

            # Example of accessing some specific parameters that might be relevant for defense
            if agent_obs['ball_owned_team'] == 1:  # Opponent has the ball
                ball_position = agent_obs['ball']
                player_position = agent_obs['right_team'][agent_obs['active']]

                # Calculating distance to ball to reward good positioning
                distance_to_ball = distance.euclidean(ball_position[:2], player_position)
                if distance_to_ball < 0.1 and self.defensive_achievement_counter[agent_idx, 0] < 1:
                    defensive_score += self.defensive_rewards[0]
                    self.defensive_achievement_counter[agent_idx, 0] = 1

                # Recognizing tackles, stops by checking change in ball ownership or direction
                if agent_obs['sticky_actions'][9]:  # Consider sliding action as a tackle attempt
                    if agent_obs['ball_owned_team'] != 1:
                        defensive_score += self.defensive_rewards[3]
                        self.defensive_achievement_counter[agent_idx, 3] += 1

            # Aggregate reward modifications
            reward[agent_idx] += defensive_score
            components["defensive_rewards"][agent_idx] = defensive_score
        
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
