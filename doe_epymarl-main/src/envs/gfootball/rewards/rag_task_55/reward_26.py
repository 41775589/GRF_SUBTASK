import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            components["tackle_reward"][rew_index] = 0.0
            
            # Conditions to modify reward for defensive actions
            if o['game_mode'] in {3, 4, 5, 6}:  # Defensive game modes
                defensive_pos = np.abs(o['left_team'][o['active']][0]) > 0.5  # Close to own goal
                has_ball = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']  # Player owns the ball

                # Enhance reward for beneficial tackles
                if o['sticky_actions'][7]:  # action_slide
                    if defensive_pos and not has_ball:
                        components["tackle_reward"][rew_index] = 0.5  # Reward for sliding tackle in a defensive position without foul
                elif o['sticky_actions'][8]:  # action_sprint -- assume good position
                    if defensive_pos:
                        components["tackle_reward"][rew_index] = 0.1  # Reward for positioning to tackle

            # Apply the rewards
            reward[rew_index] += components["tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observations, reward, done, info
