import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance team synergy during possession changes, emphasizing timing and strategic positioning."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_tracker = {}
        self.positioning_tracker = {}
        self.possession_change_reward = 0.5
        self.positioning_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_tracker = {}
        self.positioning_tracker = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'possession_change_tracker': self.possession_change_tracker,
            'positioning_tracker': self.positioning_tracker
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.possession_change_tracker = from_pickle['CheckpointRewardWrapper']['possession_change_tracker']
        self.positioning_tracker = from_pickle['CheckpointRewardWrapper']['positioning_tracker']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "possession_change_reward": [0.0] * len(reward), 
                      "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            current_possession = o['ball_owned_team']
            if current_possession != -1:
                prev_possession = self.possession_change_tracker.get(rew_index, -1)
                if prev_possession != -1 and prev_possession != current_possession:
                    components["possession_change_reward"][rew_index] = self.possession_change_reward
                    reward[rew_index] += self.possession_change_reward
                self.possession_change_tracker[rew_index] = current_possession

            player_pos = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            opponent_goal = 1 if o['ball_owned_team'] == 0 else -1
            avg_player_dist_to_goal = np.mean([abs(opponent_goal - pos[0]) for pos in player_pos])
            if avg_player_dist_to_goal < 0.3:  # Threshold for being well-positioned
                components["positioning_reward"][rew_index] = self.positioning_reward
                reward[rew_index] += self.positioning_reward
                self.positioning_tracker[rew_index] = True
            else:
                self.positioning_tracker[rew_index] = False

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
