import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that evaluates performance based on blocking and preventing opponent progress with a focus on 
    man-marking intensively, stopping attackers effectively, and blocking shots.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.opponent_progression_penalty = -0.01
        self.blocking_bonus = 0.1
        self.shot_blocking_bonus = 0.2
        self.game_mode_snapshot = -1  # Snapshotting the game mode to detect changes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.game_mode_snapshot = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter.copy()
        to_pickle['CheckpointRewardWrapper_game_mode_snapshot'] = self.game_mode_snapshot
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        self.game_mode_snapshot = from_pickle['CheckpointRewardWrapper_game_mode_snapshot']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "blocking_bonus": [0.0] * len(reward),
            "shot_blocking_bonus": [0.0] * len(reward),
            "opponent_progression_penalty": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_game_mode = o['game_mode']

            # Penalizing for opponent's progression when game mode changes showing a set play (e.g., corner, free kick)
            if current_game_mode != self.game_mode_snapshot and current_game_mode != 0:
                components["opponent_progression_penalty"][rew_index] = self.opponent_progression_penalty
                reward[rew_index] += components["opponent_progression_penalty"][rew_index]

            # Reward for man-marking and blocking if an opponent with the ball was stopped
            if 'opponent_active' in o and o['opponent_active'] and not o.get('ball_owned_team', 1) == 1:
                components["blocking_bonus"][rew_index] = self.blocking_bonus
                reward[rew_index] += components["blocking_bonus"][rew_index]

            # Additional reward for blocking shots
            if current_game_mode in (5, 6):  # Assuming modes for penalty and free-kick dangerously close
                components["shot_blocking_bonus"][rew_index] = self.shot_blocking_bonus
                reward[rew_index] += components["shot_blocking_bonus"][rew_index]

            # Update the snapshot of last game mode
            self.game_mode_snapshot = current_game_mode

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
