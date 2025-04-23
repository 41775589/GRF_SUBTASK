import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering sliding tackles during counter-attacks and high-pressure defensive actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Base reward preserved
            components['base_score_reward'][rew_index] = reward[rew_index]
            
            # Check game modes relevant for counters and defensive high-pressure
            game_mode = o['game_mode']
            if game_mode == 2 or game_mode == 4:
                # Further checks on positioning and ball possession
                if o['ball_owned_team'] != self.env.unwrapped.team_id:
                    # Check position to encourage tackles when the opponent is near the defense third
                    own_team_positions = o['left_team'] if self.env.unwrapped.team_id == 0 else o['right_team']
                    opponent_positions = o['right_team'] if self.env.unwrapped.team_id == 0 else o['left_team']
                    defense_third_x = 0.33 if self.env.unwrapped.team_id == 0 else -0.33
                    close_opponents = np.any(opponent_positions[:, 0] > defense_third_x if self.env.unwrapped.team_id == 0 else opponent_positions[:, 0] < defense_third_x)
                    
                    if close_opponents:
                        # Encourage sliding tackle (action index 9 corresponds to sliding)
                        if o['sticky_actions'][9]:
                            components['tackle_reward'][rew_index] = 0.2
                            reward[rew_index] += components['tackle_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
