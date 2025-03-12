import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive strategies including shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward = 0.1
        self.shoot_reward = 0.2
        self.dribble_reward = 0.05

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
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for passing effectiveness based on sticky actions
            if 'sticky_actions' in o and o['sticky_actions'][6]:  # Typically, high or long pass action
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

            # Reward for shooting towards the goal
            if 'game_mode' in o and o['game_mode'] == 6:  # Assuming mode 6 corresponds to a shot or attempt to score
                components["shoot_reward"][rew_index] = self.shoot_reward
                reward[rew_index] += components["shoot_reward"][rew_index]

            # Reward for dribbling based on ball possession and movement
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active'] and 'ball_direction' in o:
                if np.linalg.norm(o['ball_direction'][:2]) > 0.01:  # Ball is moving and owned by the player's team
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
