import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for specialized defensive training tasks:
    - Rewards for successful tackling
    - Rewards for efficient stopping near opponents
    - Rewards for executing successful pressured passes
    This encourages players to learn defensive strategies.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "stopping_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        tackle_coefficient = 0.2
        stopping_coefficient = 0.1
        passing_coefficient = 0.3

        for rew_index, o in enumerate(observation):
            active_player_pos = o['left_team'][o['active']]
            active_player_opponents = o['right_team']
            
            # Tackling: Gives reward if the active player is closer than others and decrease opponent's ball possession
            if o['ball_owned_team'] == 1:
                distances_to_ball = np.linalg.norm(active_player_opponents - o['ball'][:2], axis=1)
                if np.min(distances_to_ball) > np.linalg.norm(o['ball'][:2] - active_player_pos):
                    components["tackle_reward"][rew_index] = tackle_coefficient

            # Stopping Efficiently: Reward for stopping close to an opponent who has the ball
            if o['ball_owned_team'] == 1:
                opponent_with_ball = active_player_opponents[o['ball_owned_player']]
                distance_to_opponent_with_ball = np.linalg.norm(opponent_with_ball - active_player_pos)
                if distance_to_opponent_with_ball < 0.1:
                    components["stopping_reward"][rew_index] = stopping_coefficient

            # Pressured passing: Reward for passing successfully when surrounded by opponents
            if o['sticky_actions'][9]:  # Assuming index 9 is related to some kind of pass action
                num_close_opponents = np.sum(np.linalg.norm(active_player_opponents - active_player_pos, axis=1) < 0.2)
                if num_close_opponents >= 2:
                    components["pressured_pass_reward"][rew_index] = passing_coefficient

            # Calculate total reward based on added components
            reward[rew_index] += (components["tackle_reward"][rew_index] + 
                                  components["stopping_reward"][rew_index] +
                                  components["pressured_pass_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = max(self.sticky_actions_counter[i], action)
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
