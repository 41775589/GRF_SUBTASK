import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering offensive tactics in football."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Assuming goals are at x=1 (right side of the field)
        self._goal_x = 1

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        new_rewards = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage movement towards the opponent's goal and successful passes or shots
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Team 0 is the agent's team
                if o['ball_owned_player'] == o['active']:  # Ball is owned by the active player
                    # Direction to goal (assuming goal is at x=1 and on the horizontal axis)
                    target_direction = np.array([self._goal_x, o['ball'][1]]) - o['ball'][:2]
                    target_direction /= np.linalg.norm(target_direction)
                    # Current movement direction
                    current_direction = np.array(o['ball_direction'])
                    
                    # Reward alignment of ball movement towards the goal
                    directional_reward = np.dot(target_direction, current_direction)
                    new_rewards[rew_index] += 0.1 * directional_reward

                    # Rewards for successful actions
                    if 'sticky_actions' in o:
                        if o['sticky_actions'][7] == 1 or o['sticky_actions'][8] == 1:  # Short Pass or Long Pass
                            new_rewards[rew_index] += 0.2
                        if o['sticky_actions'][9] == 1:  # Shot
                            new_rewards[rew_index] += 0.3

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
