import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards strategic ball positioning for shooting and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_reached = np.zeros(3)  # Assuming three checkpoints: for passing, long shoot, strategic position.
        self.passing_reward = 0.1
        self.shooting_reward = 0.2
        self.position_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_reached = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_reached
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_reached = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            player_obs = observation[idx]
            ball_pos = np.array(player_obs['ball'])
            ball_owning = player_obs['ball_owned_team'] == 0

            # Conditions based on predefined game strategy
            close_to_goal = ball_pos[0] > 0.8  # Ball is close to opponent's goal
            close_to_midfield = np.abs(ball_pos[0]) < 0.2  # Ball is around the midfield
            strategic_position = np.abs(ball_pos[1]) < 0.1  # Ball is centrally positioned

            if ball_owning and close_to_goal and self.checkpoint_reached[idx] < 1:
                reward[idx] += self.shooting_reward
                components['shooting_reward'][idx] = self.shooting_reward
                self.checkpoint_reached[idx] = 1

            if ball_owning and close_to_midfield and self.checkpoint_reached[idx] < 1:
                reward[idx] += self.passing_reward
                components['passing_reward'][idx] = self.passing_reward
                self.checkpoint_reached[idx] = 1
            
            if ball_owning and strategic_position and self.checkpoint_reached[idx] < 1:
                reward[idx] += self.position_reward
                components['position_reward'][idx] = self.position_reward
                self.checkpoint_reached[idx] = 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
