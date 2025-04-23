import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized finishing technique rewards, focusing on shooting angles and timing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_regions = np.linspace(-0.42, 0.42, 10)  # Divide Y-axis in front of the goal into sections
        self.shooting_rewards = np.linspace(0.1, 1.0, 10)  # Reward increased with proximity to the center of the goal
        self.pressure_multiplier = 1.5  # Increase rewards when shooting under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, rewards):
        rewards_list = list(rewards)
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards_list.copy()}

        for i in range(len(rewards_list)):
            # Extract necessary observation parameters
            o = observation[i]
            ball_pos = o['ball'][:2]
            player_pos = o['right_team'][o['active']]
            goal_y_range = [-0.044, 0.044]  # goal Y position range
            
            # Check if the ball is close to the opponent's goal and if it is under control
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                distance_from_goal = 1 - ball_pos[0]  # Normalized X distance from goal (right side for the opponent)
                if distance_from_goal < 0.1:  # Shoots are taken near the goal
                    # Calculate Y deviation from the goal center and assign a reward based on closeness to the middle
                    y_deviation = np.abs(ball_pos[1])
                    region_index = np.digitize(y_deviation, self.shooting_regions) - 1
                    region_reward = self.shooting_rewards[region_index] * self.pressure_multiplier if o['game_mode'] != 0 else self.shooting_rewards[region_index]
                    rewards_list[i] += region_reward
                    # Track reward components
                    components[f'shooting_reward_p{i}'] = region_reward

        return rewards_list, components

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
