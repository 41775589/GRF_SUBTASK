import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive actions by the goalkeeper and defenders."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize checkpoint counts and rewards for various defensive actions
        self.goalkeeper_saves = 0
        self.defender_tackles = 0
        self.play_initiations = 0
        self.tackle_reward = 0.3
        self.save_reward = 0.5
        self.initiation_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_saves = 0
        self.defender_tackles = 0
        self.play_initiations = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'goalkeeper_saves': self.goalkeeper_saves,
            'defender_tackles': self.defender_tackles,
            'play_initiations': self.play_initiations
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        data = from_pickle['CheckpointRewardWrapper']
        self.goalkeeper_saves = data['goalkeeper_saves']
        self.defender_tackles = data['defender_tackles']
        self.play_initiations = data['play_initiations']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "save_reward": [0.0] * len(reward),
            "initiation_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if goalie made a save
            if o['active_role'] == e_PlayerRole_GK and o['ball_owned_team'] == 0 and o['ball_close']:
                components["save_reward"][rew_index] = self.save_reward
                reward[rew_index] += components["save_reward"][rew_index]
                self.goalkeeper_saves += 1
            
            # Check if defenders made a tackle
            if o['active_role'] in [e_PlayerRole_CB, e_PlayerRole_LB, e_PlayerRole_RB] and o['ball_owned_team'] == 0 and o['tackle']:
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
                self.defender_tackles += 1

            # Reward for initiating a play
            if o['active_role'] in [e_PlayerRole_CB, e_PlayerRole_LB, e_PlayerRole_RB, e_PlayerRole_GK] and o['initiate_play']:
                components["initiation_reward"][rew_index] = self.initiation_reward
                reward[rew_index] += components["initiation_reward"][rew_index]
                self.play_initiations += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
