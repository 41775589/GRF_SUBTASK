import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward mechanism focusing 
    on mastering offensive strategies including accurate shooting, 
    effective dribbling, and practicing long/high passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribbling_bonus = 0.05
        self.pass_bonus = 0.03
        self.shoot_bonus = 0.1

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Start with the base score reward from the parent environment.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "dribbling_bonus": [0.0] * len(reward),
                      "pass_bonus": [0.0] * len(reward),
                      "shoot_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_set = o['sticky_actions']

            # Assess dribbling (dribble and move actions combined)
            if action_set[9] == 1 and (any(action_set[0:8])):
                components["dribbling_bonus"][rew_index] = self.dribbling_bonus
                reward[rew_index] += components["dribbling_bonus"][rew_index]

            # Assess passes (long and high pass assessment)
            if action_set[8] == 1 and (action_set[6] == 1 or action_set[7] == 1):
                components["pass_bonus"][rew_index] = self.pass_bonus
                reward[rew_index] += components["pass_bonus"][rew_index]

            # Boost shooting towards goal
            if o['game_mode'] == 6 and action_set[2]:  # Assuming mode 6 is a shooting scenario
                components["shoot_bonus"][rew_index] = self.shoot_bonus
                reward[rew_index] += components["shoot_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add each reward components to info dictionary for detailed logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
