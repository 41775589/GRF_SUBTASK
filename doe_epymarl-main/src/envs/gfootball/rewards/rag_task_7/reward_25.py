import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering defensive maneuvers, specifically sliding tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_tackles_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_reward = 0.05
        self.sliding_tackle_threshold = 3

    def reset(self):
        self.sliding_tackles_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sliding_tackles_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.sliding_tackles_counter = from_picle['CheckpointRewardWrapper']
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_tackle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["sliding_tackle_reward"][rew_index] = 0.0

            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Defensive game modes
                active_player = o['active']
                ball_pos = o['ball'][:2]  # Ignore z-axis
                player_pos = o['left_team'][active_player] if o['ball_owned_team'] == 0 else o['right_team'][active_player]
                
                distance_to_ball = np.linalg.norm(ball_pos - player_pos)
                
                if distance_to_ball < 0.02:  # Close to the ball
                    self.sliding_tackles_counter[rew_index] += 1
                    if self.sliding_tackles_counter[rew_index] <= self.sliding_tackle_threshold:
                        components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward
                    
            reward[rew_index] += components["sliding_tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
