import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive maneuvers and game phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_maneuver_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["offensive_maneuver_reward"][rew_index] = 0.0

            # Check game mode: is it a normal play or a special situation (like corners, free kicks)
            if o['game_mode'] == 0:  # Normal gameplay
                # Encourage forward movements with the ball
                if o['ball_owned_team'] == 0:  # if the left team (controlled by agent) has the ball
                    ball_x = o['ball'][0]
                    player_x = o['left_team'][o['active']][0]  # x position of the controlled player
                    if ball_x > 0.5 and player_x > 0.5:  # progressing towards opponent's goal
                        components["offensive_maneuver_reward"][rew_index] += 0.1
                        if o['ball_owned_player'] == o['active']:
                            components["offensive_maneuver_reward"][rew_index] += 0.2  # possession bonus

            # Encourage scoring
            if reward[rew_index] == 1:  # goal scored
                components["offensive_maneuver_reward"][rew_index] += 1.0
        
        # Update rewards with weighted components
        for rew_index in range(len(reward)):
            reward[rew_index] += components["offensive_maneuver_reward"][rew_index]

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
