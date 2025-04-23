import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper designed to emphasize collaborative plays between shooters (players near the opponent's goal) and passers.
    This is achieved by rewarding pass completions leading to attempts on goal.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize variables to track the state of player possessions and shots 
        self.pass_received_near_goal = {}
        # A threshold for defining 'near goal' area
        self.near_goal_threshold = 0.7
        self.pass_reward = 0.2

    def reset(self):
        self.pass_received_near_goal = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_received_near_goal
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_received_near_goal = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaborative_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Base reward stays unchanged; the focus is on the collaborative component
            components["base_score_reward"][rew_index] = reward[rew_index]
            ball_owned_team = o.get('ball_owned_team', -1)

            if ball_owned_team == 0 and 'ball_owned_player' in o and 'right_team' in o:
                # Check if ball is near the opponent's goal
                ball_position = o.get('ball', [0, 0])
                if abs(ball_position[0]) > self.near_goal_threshold:
                    # Player has ball near the opponent's goal
                    # Check if the ball was passed to them
                    if o['ball_owned_player'] not in self.pass_received_near_goal:
                        self.pass_received_near_goal[o['ball_owned_player']] = True
                        components["collaborative_reward"][rew_index] = self.pass_reward
                        reward[rew_index] += self.pass_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
