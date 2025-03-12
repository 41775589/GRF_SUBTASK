import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This class adds a reward aimed at developing offensive strategies. It includes rewards for:
    1. Successful dribbling maneuvers towards the opposing goal.
    2. Accurate shooting attempts.
    3. Effective passes that break through the opponent's defensive lines.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribbling_bonus = 0.1
        self.shooting_bonus = 0.5
        self.passing_bonus = 0.3
    
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
        components = {"base_score_reward": np.copy(reward),
                      "dribbling_bonus": [0.0] * len(reward),
                      "shooting_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == o['designated']:
                distance_to_goal = np.abs(o['ball'][0] - 1)

                # Dribbling Towards Goal
                if o['sticky_actions'][8] == 1:  # Dribble action is active
                    components["dribbling_bonus"][i] += self.dribbling_bonus * distance_to_goal

                # Shooting Towards Goal
                if o['game_mode'] in [3, 6]:  # in a FreeKick or Penalty mode
                    components["shooting_bonus"][i] += self.shooting_bonus

                # Effective Passing
                if o['ball_direction'][0] > 0:  # Ball is moving towards opponent's goal
                    components["passing_bonus"][i] += self.passing_bonus

            # Total reward update incorporating bonuses
            reward[i] += (components["dribbling_bonus"][i] +
                          components["shooting_bonus"][i] +
                          components["passing_bonus"][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
