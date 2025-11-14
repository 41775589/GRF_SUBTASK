import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on ball control and passing efficiency for a specific player."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "control_dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Encourage maintaining ball possession and executing successful passes
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                ball_control = 0.1  # reward for controlling the ball
                
                # Check if dribble action is used effectively
                if 'sticky_actions' in o and o['sticky_actions'][9]:  # index 9 corresponds to action_dribble
                    components["control_dribble_reward"][rew_index] += ball_control
                    reward[rew_index] += 0.5 * components["control_dribble_reward"][rew_index]

                # Reward for successful passes
                if 'game_mode' in o and o['game_mode'] in [1, 2, 3, 4]:  # Checking if a pass leads to a game restart mode
                    pass_effectiveness = 0.3
                    components["control_dribble_reward"][rew_index] += pass_effectiveness
                    reward[rew_index] += 0.5 * components["control_dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
