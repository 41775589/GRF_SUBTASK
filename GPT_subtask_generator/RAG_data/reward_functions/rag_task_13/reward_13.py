import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on 'stopper' behaviors."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocked_shots = 0
        self.interceptions = 0
        self.defensive_actions = 0
        self.position_importance = 0.05  # Increase or decrease importance based on testing

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocked_shots = 0
        self.interceptions = 0
        self.defensive_actions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['blocked_shots'] = self.blocked_shots
        to_pickle['interceptions'] = self.interceptions
        to_pickle['defensive_actions'] = self.defensive_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.blocked_shots = from_pickle['blocked_shots']
        self.interceptions = from_pickle['interceptions']
        self.defensive_actions = from_pickle['defensive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_reward"][rew_index] = 0.0
            
            if o['ball_owned_team'] == 1: # If ball is owned by the opponent
                defensive_positioning = np.abs(o['left_team'][o['active']][0])
                components["defensive_reward"][rew_index] = defensive_positioning * self.position_importance
                reward[rew_index] += components["defensive_reward"][rew_index]
                
            # Count interceptions and blocks as additional rewards
            if o['game_mode'] == 3 and o['ball_owned_team'] == 0: # Freekick interception
                self.interceptions += 1
                reward[rew_index] += 1.0
            
            # Reward blocking opponent actions specifically
            if o['ball_owned_team'] == 1 and o['right_team'][o['ball_owned_player']][0] < 0.5 and np.abs(o['ball_direction'][0]) > 0.2:
                # This assumes the opponent is trying to forward the ball to score
                self.blocked_shots += 1
                reward[rew_index] += 1.0

            # General defensive action rewards
            current_actions = np.nonzero(o['sticky_actions'])[0]
            if 9 in current_actions: # Checking if dribble or aggressive actions are blocked
                self.defensive_actions += 1
                reward[rew_index] += 0.3

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
