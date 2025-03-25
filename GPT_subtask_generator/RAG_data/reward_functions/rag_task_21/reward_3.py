import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering defensive responsiveness."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.player_position = {}
        self._interception_reward = 1.0
        self._defensive_position_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_position = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.player_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.player_position = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "defensive_position_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            ball_owned_team = obs.get('ball_owned_team', -1)
            active_player_pos = obs['left_team'][obs['active']] if ball_owned_team == 1 else obs['right_team'][obs['active']]
            ball_position = obs['ball'][:2]

            # Reward for interception: Increase when gaining ball possession.
            if ball_owned_team != -1 and (ball_owned_team == 0 and obs['ball_owned_player'] == obs['active']):
                components['interception_reward'][i] = self._interception_reward

            # Reward for maintaining healthy defensive positioning.
            if self.env.unwrapped.observation()['game_mode'] in [3, 4]:  # Free kick or Corner
                defensive_pos = self.defensive_rating(active_player_pos, ball_position)
                components['defensive_position_reward'][i] = defensive_pos * self._defensive_position_reward

            # Update reward list
            reward[i] += components['interception_reward'][i] + components['defensive_position_reward'][i]

        return reward, components

    def defensive_rating(self, player_pos, ball_pos):
        """Calculate a simple defensive rating based on the distance from the ball."""
        dist = np.linalg.norm(np.array(player_pos) - np.array(ball_pos))
        rating = max(0, 1 - dist / 1.5)  # assumes distance is normalized in some meaningful way
        return rating

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
