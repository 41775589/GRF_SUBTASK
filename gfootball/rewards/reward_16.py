import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        self._possession_rewards = {}
        self._num_pass_checkpoints = 5
        self._pass_reward = 0.1
        self._shooting_manual_reward = 0.5
        self._ball_advanced_rewards = {}
        self._dribbling_advanced_reward = 0.05

    def reset(self):
        self._possession_rewards = {}
        self._ball_advanced_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['offensive_strategy_rewards'] = (self._possession_rewards, self._ball_advanced_rewards)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._possession_rewards, self._ball_advanced_rewards = from_pickle['offensive_strategy_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            current_reward = 0
            did_score = reward[i] == 1
            ball_owned_by_this_team = o['ball_owned_team'] == 0 if 'ball_owned_team' in o else False
            player_with_ball = o['active'] if 'ball_owned_player' in o and o['ball_owned_player'] >= 0 else -1

            # Passing progression
            if ball_owned_by_this_team:
                num_passes = self._possession_rewards.get(i, 0)
                if num_passes < self._num_pass_checkpoints:
                    current_reward += self._pass_reward 
                    self._possession_rewards[i] = num_passes + 1
                components["passing_reward"][i] = current_reward

            # Shooting effectively adds a manual bonus
            if did_score:
                current_reward += self._shooting_manual_reward
                components["shooting_reward"][i] = self._shooting_manual_reward

            # Reward for advancing the ball forward while retaining possession
            if player_with_ball != -1:
                ball_position = o['ball'][0]
                previous_position = self._ball_advanced_rewards.get(i, -1)
                if ball_position > previous_position:
                    dribble_advantage = ball_position - previous_position
                    dribble_reward = dribble_advantage * self._dribbling_advanced_reward  # Calculate proportional reward
                    current_reward += dribble_reward
                    components["dribbling_reward"][i] = dribble_reward
                self._ball_advanced_rewards[i] = ball_position
                
            reward[i] += current_reward

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and add each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
