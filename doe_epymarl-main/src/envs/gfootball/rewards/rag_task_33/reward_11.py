import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes shooting from distance and beating defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distance thresholds for long-range shooting
        self.long_shot_threshold = 0.6  # Threshold for considering a shot as long-range
        self.shot_distance_reward = 0.3  # Reward for taking a shot from long range
        self.defender_beating_reward = 0.2  # Reward for dribbling past a defender
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        return self.env.set_state(state)
        
    def reward(self, reward):
        """Computes augmented reward based on long-distance shot and beating defenders."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "long_shot_reward": [0.0] * len(reward),
                      "defender_beating_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculating distance of the ball from the opponent's goal
            opponent_goal_x = 1
            ball_x, ball_y = o['ball'][0], o['ball'][1]
            distance = np.abs(opponent_goal_x - ball_x)

            # Checking if a long shot was taken
            if distance >= self.long_shot_threshold:
                components['long_shot_reward'][rew_index] = self.shot_distance_reward
                reward[rew_index] += self.shot_distance_reward

            # Count the number of opponents closer to the goal than the player
            opponent_positions = o['right_team']
            player_x, player_y = o['left_team'][o['active']][0], o['left_team'][o['active']][1]
            num_defenders = sum([1 for pos in opponent_positions if (pos[0] > player_x)])

            # Reward for dribbling past defenders
            if num_defenders < 2:
                components['defender_beating_reward'][rew_index] = self.defender_beating_reward
                reward[rew_index] += self.defender_beating_reward

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
