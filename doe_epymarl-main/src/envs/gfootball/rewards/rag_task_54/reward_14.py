import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for collaborative plays between shooters and passers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_attempt_count = 0
        self.pass_complete_count = 0
        self.collaborative_play_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_attempt_count = 0
        self.pass_complete_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shot_attempt_count': self.shot_attempt_count,
            'pass_complete_count': self.pass_complete_count
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_attempt_count = from_pickle['CheckpointRewardWrapper']['shot_attempt_count']
        self.pass_complete_count = from_pickle['CheckpointRewardWrapper']['pass_complete_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaborative_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # If any goal is scored, we check for the recent collaboration:
            if 'score' in o and o['score'] != components['base_score_reward']:
                components["collaborative_play_reward"][rew_index] = self.collaborative_play_reward * \
                                                                      (self.shot_attempt_count + self.pass_complete_count)
                reward[rew_index] += components["collaborative_play_reward"][rew_index]
                self.reset_counters()  # reset counters after rewarding for a goal
            
            # Checking for shots and passes:
            if o['ball_owned_team'] == 1 and 'ball_owned_player' in o:
                if self.is_shot_attempt(o):
                    self.shot_attempt_count += 1
                if self.is_pass_complete(o):
                    self.pass_complete_count += 1
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def is_shot_attempt(self, observation):
        # Custom logic to determine a shot attempt, traditionally involves high ball direction towards goal
        return 'ball_direction' in observation and abs(observation['ball_direction'][1]) > 0.5
    
    def is_pass_complete(self, observation):
        # Custom logic to determine a pass complete, usually movement of the ball towards another player
        return 'ball_owned_team' in observation and observation['ball_owned_team'] == observation['active']
        
    def reset_counters(self):
        self.shot_attempt_count = 0
        self.pass_complete_count = 0
