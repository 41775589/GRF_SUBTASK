import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes scoring through high passes and crossing from different angles and distances.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_player_ball = None  # Tracks the last player who made a successful high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_player_ball = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_player_ball': self.previous_player_ball
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_player_ball = from_pickle['CheckpointRewardWrapper']['previous_player_ball']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # High pass bonus: If ball_owned_team is the current team and if the pass height is above a threshold
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:    # Assuming 0 is the team index of `self`
                ball_z = o['ball'][2]
                if ball_z > 0.15:  # Threshold for considering it a 'high' pass
                    last_player = o['ball_owned_player']
                    
                    # Reward for high pass if it's not by the last player who did a high pass
                    if last_player != self.previous_player_ball:
                        components["high_pass_reward"][rew_index] = 0.5
                        reward[rew_index] += components["high_pass_reward"][rew_index]
                        self.previous_player_ball = last_player

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
