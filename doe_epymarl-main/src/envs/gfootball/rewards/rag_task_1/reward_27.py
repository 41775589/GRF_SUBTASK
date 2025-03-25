import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to promote offensive plays by rewarding fast break attacks and adaptive behaviors in different game modes.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_checkpoints = 5  # Number of regions to control before a fast break is rewarded
        self.checkpoint_reward = 0.2
        self.fast_break_bonus = 1.0
        self.dynamic_game_mode_reward = 0.5
        self.previous_ball_holder = None
        self.ball_progression = 0
        self.dynamic_adaptation = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_holder = None
        self.ball_progression = 0
        self.dynamic_adaptation = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward),
                      "fast_break_bonus": [0.0] * len(reward),
                      "dynamic_game_mode_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check current ball possession and location
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] != -1:
                self.dynamic_adaptation |= o['game_mode'] != 0
                # Determine distance to goal for fast break evaluation
                ball_position = o['ball'][0]  # Only consider x-axis for simplicity in this example
                if self.previous_ball_holder is None or self.previous_ball_holder != o['ball_owned_player']:
                    self.previous_ball_holder = o['ball_owned_player']
                    self.ball_progression = ball_position
               
                if ball_position > self.ball_progression:
                    self.ball_progression = ball_position
                    distance_covered = ball_position - self.ball_progression

                    if distance_covered >= self.num_checkpoints:
                        components["fast_break_bonus"][rew_index] = self.fast_break_bonus
                        reward[rew_index] += components["fast_break_bonus"][rew_index]
                        
                if self.dynamic_adaptation:
                    components["dynamic_game_mode_bonus"][rew_index] = self.dynamic_game_mode_reward
                    reward[rew_index] += components["dynamic_game_mode_bonus"][rew_index]
                    
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
