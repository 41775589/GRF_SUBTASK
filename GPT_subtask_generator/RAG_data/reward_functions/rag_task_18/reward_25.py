import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for effective midfield play focusing on transitions and pace management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards = [0.0, 0.0, 0.0]  # Rewards for each agent
        self.pace_rewards = [0.0, 0.0, 0.0]  # Rewards for each agent

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards = [0.0, 0.0, 0.0]
        self.pace_rewards = [0.0, 0.0, 0.0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'transition_rewards': self.transition_rewards,
            'pace_rewards': self.pace_rewards
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = state_data['sticky_actions_counter']
        self.transition_rewards = state_data['transition_rewards']
        self.pace_rewards = state_data['pace_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [], "pace_reward": []}
        
        if observation is None:
            return reward, components

        components["transition_reward"] = self.transition_rewards.copy()
        components["pace_reward"] = self.pace_rewards.copy()

        for i, o in enumerate(observation):
            ball_control = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']
            if ball_control:
                # Improvement in movement towards the opposition's half
                dx = o['left_team_direction'][o['active']][0]  # Positive x-direction is good
                if dx > 0:
                    self.transition_rewards[i] += 0.1 * dx  # Reward for moving forward
                components["transition_reward"][i] = self.transition_rewards[i]

                # Reward for maintaining control under pressure
                pace = np.linalg.norm(o['left_team_direction'][o['active']])
                if pace < 0.01:  # rewarded for controlled, slow-paced movement
                    self.pace_rewards[i] += 0.1 
                    
                components["pace_reward"][i] = self.pace_rewards[i]

            # Calculate the combined reward
            total_transition_reward = sum(self.transition_rewards)
            total_pace_reward = sum(self.pace_rewards)
            reward[i] += total_transition_reward + total_pace_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
