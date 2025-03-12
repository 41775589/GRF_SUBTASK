import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on developing offensive strategies.
    It incentivizes accurate shooting, effective dribbling, and mastering different types of passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_possession = None

    def reset(self):
        self.ball_possession = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_possession'] = self.ball_possession
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_possession = from_pickle.get('ball_possession', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "shooting_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward),
                      "pass_bonus": [0.0] * len(reward)}

        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']: 
                if self.ball_possession != o['active']:
                    # Ball possession has changed, assuming a successful pass.
                    components["pass_bonus"][i] = 0.1
                self.ball_possession = o['active']
        
            # Bonus for dribbling: increasing if player is controlling ball and moving
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                components["dribble_bonus"][i] = 0.05 * (abs(o['right_team_direction'][o['active']]).sum())

            # Bonus for shooting: check ball direction towards goal when shooting is attempted
            if o['game_mode'] == 6 and o['ball_direction'][0] > 0:  # Simplified assumption for shot direction
                components["shooting_bonus"][i] = 0.3

        # Update rewards
        for i in range(len(reward)):
            reward[i] += (components["shooting_bonus"][i] + 
                        components["dribble_bonus"][i] + 
                        components["pass_bonus"][i])
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
