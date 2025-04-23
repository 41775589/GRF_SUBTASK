import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that distributes rewards based on midfielders' effective passing in the offensive phase and strikers scoring goals."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter
        self._midfield_pass_reward = 0.2
        self._striker_score_reward = 1.0
        self._midfield_positions_max = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._midfield_positions_max = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'midfield_positions_max': self._midfield_positions_max
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._midfield_positions_max = from_pickle['CheckpointRewardWrapper']['midfield_positions_max']
        return from_pickle

    def reward(self, reward):
        original_reward = reward.copy()  # Keep track of the initial reward
        components = {
            "base_score_reward": reward.copy(),
            "midfield_pass_reward": [0.0] * len(reward),
            "striker_score_reward": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if 'ball_owned_team' not in obs or obs['ball_owned_team'] != 0:
                continue

            current_player_role = obs['left_team_roles'][obs['active']]
            
            # High reward for midfielders making forward passes
            if current_player_role in [4, 5, 6]:  # DM, CM, LM roles
                ball_position = obs['ball'][0]  # Check only x position
                max_position_reached = self._midfield_positions_max.get(i, -1)
                if ball_position > max_position_reached:
                    self._midfield_positions_max[i] = ball_position
                    components['midfield_pass_reward'][i] = self._midfield_pass_reward
                    reward[i] += components['midfield_pass_reward'][i]

            # Add rewards for strikers when a goal is scored
            if current_player_role == 9:  # Striker: CF
                if reward[i] > 0:  # Initially provided by the env for scoring
                    components['striker_score_reward'][i] = self._striker_score_reward
                    reward[i] += components['striker_score_reward'][i]

        # Returns the modified rewards along with the breakdown of each component
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
