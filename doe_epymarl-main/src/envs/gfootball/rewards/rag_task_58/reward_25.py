import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward based on defensive play and transitioning to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._opponent_position_threshold = 0.5
        self._transition_reward = 0.2
        self._defense_reward = 0.1
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper'].get('last_ball_position')
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defense_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_position = np.array(o['ball'][:2])  # Get the ball's x, y coordinates

            # Encourage defense by checking if the ball is closer to the own goal
            opponent_team_positions = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
            ball_close_to_opponent = np.any([
                np.linalg.norm(ball_position - player_position) < self._opponent_position_threshold
                for player_position in opponent_team_positions
            ])

            if ball_close_to_opponent:
                components["defense_reward"][rew_index] = self._defense_reward
                reward[rew_index] += components["defense_reward"][rew_index]

            # Reward for transition from defense to attack through ball control
            if self.last_ball_position is not None:
                ball_movement = np.linalg.norm(ball_position - self.last_ball_position)
                own_team_has_ball = o['ball_owned_team'] == 1 if o['ball_owned_team'] != -1 else False
                if ball_movement > 0.05 and own_team_has_ball:
                    components["transition_reward"][rew_index] = self._transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]

            self.last_ball_position = ball_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
