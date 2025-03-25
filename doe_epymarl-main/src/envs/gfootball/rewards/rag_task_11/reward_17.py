import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive maneuvers and finishing accuracy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds and reward levels for different stages of offensive play
        self.field_zones_thresholds = np.linspace(-1, 1, num=11)  # Divide the field into 10 zones
        self.zone_rewards = np.linspace(0.1, 0.5, num=10)  # Increasing rewards as the agent approaches the goal
        self.scoring_zone_threshold = 0.8  # Close to the opponent's goal
        self.scoring_zone_reward = 1.0     # Reward for operating in the scoring zone
        self.goal_reward = 5.0  # High reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x_position = o['ball'][0]
            
            # Reward for advancing the ball in different field zones
            for zone_index, threshold in enumerate(self.field_zones_thresholds[:-1]):
                if threshold <= ball_x_position < self.field_zones_thresholds[zone_index + 1]:
                    components["offensive_reward"][rew_index] += self.zone_rewards[zone_index]
                    break
            
            # Additional reward for ball control in the scoring zone
            if ball_x_position > self.scoring_zone_threshold:
                if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and
                    'ball_owned_player' in o and o['ball_owned_player'] == o['active']):
                    components["offensive_reward"][rew_index] += self.scoring_zone_reward
            
            # Additional reward for scoring a goal
            if o['game_mode'] == 6:  # Assuming game_mode 6 corresponds to a goal
                components["offensive_reward"][rew_index] += self.goal_reward
            
            reward[rew_index] += components["offensive_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
