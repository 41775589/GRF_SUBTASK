import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive strategies and maintaining ball control.
    
    The reward function will emphasize:
    - Maintaining possession of the ball, especially in the team's half of the field
    - Strategic defensive positioning near own goal
    - Efficient ball retrieval and interceptions in the defensive half
    """

    def __init__(self, env):
        super().__init__(env)
        self.current_score = {'left_team': 0, 'right_team': 0}
        self.ball_location_penalty = -0.1

    def reset(self):
        """Reset the environment state and needed variables."""
        self.current_score = {'left_team': 0, 'right_team': 0}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize wrapper state for save."""
        to_pickle['CheckpointRewardWrapper'] = {
            'current_score': self.current_score,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize wrapper state for load."""
        from_pickle = self.env.set_state(state)
        self.current_score = from_pickle['CheckpointRewardWrapper']['current_score']
        return from_pickle

    def reward(self, reward):
        """Augment reward with custom logic for better defensive play and ball control rewards."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        new_rewards = []
        for idx, o in enumerate(observation):
            # Simple reward for maintaining possession in the defensive half
            if o['ball_owned_team'] == 0 and o['ball'][0] < 0:
                reward[idx] += 0.2

            # Punish loss of possession in the defensive half
            if o['ball_owned_team'] != 0 and o['ball'][0] < 0:
                reward[idx] += self.ball_location_penalty

            # Reward for successful interception in defensive half
            if o['game_mode'] == 4 and o['ball'][0] < 0:  # 4 = Change in possession
                reward[idx] += 0.5

            new_rewards.append(reward[idx])

        components['penalties_and_bonuses'] = new_rewards
        return new_rewards, components

    def step(self, action):
        """Execute environment step and augment reward and info with custom logic."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
