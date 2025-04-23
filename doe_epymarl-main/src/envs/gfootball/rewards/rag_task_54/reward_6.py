import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the training by focusing on collaborative plays between shooters and passers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients
        self.pass_reward = 0.05
        self.shoot_reward = 0.1
        # Internal state to track last player who possessed the ball
        self.last_player_with_ball = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_player_with_ball = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_last_player_with_ball'] = self.last_player_with_ball
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_player_with_ball = from_pickle.get('CheckpointRewardWrapper_last_player_with_ball', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:  # Assuming '0' is our team
                current_player = obs['ball_owned_player']
                if current_player != self.last_player_with_ball:
                    if self.last_player_with_ball is not None:
                        # A pass has occurred
                        components['pass_reward'][i] = self.pass_reward
                        reward[i] += components['pass_reward'][i]
                    self.last_player_with_ball = current_player

            if obs['right_team'][obs['active']][0] > 0.7 and obs['sticky_actions'][4]:
                # Player is in a shooting position and tries to shoot (action index 4 is assumed to be shooting)
                components['shoot_reward'][i] = self.shoot_reward
                reward[i] += components['shoot_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_flag

        return observation, reward, done, info
