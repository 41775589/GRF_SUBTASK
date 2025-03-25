import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a complex reward based on offensive strategies, encouraging team coordination and goal scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.shoot_reward = 0.1
        self.positioning_reward = 0.01

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Base game score
            components["base_score_reward"][rew_index] += reward[rew_index]
            
            # Check if the player has the ball and is performing a shooting or passing action
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if self.sticky_actions_counter[9] == 1:  # Dribble action
                    reward[rew_index] += self.pass_reward
                    components["pass_reward"][rew_index] += self.pass_reward
                if self.sticky_actions_counter[5]:  # Bottom-right action, simulating shots
                    reward[rew_index] += self.shoot_reward
                    components["shoot_reward"][rew_index] += self.shoot_reward

            # Reward for good positioning relative to the ball position
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball']
            distance_to_ball = np.linalg.norm(player_pos - ball_pos[:2])
            positioning_bonus = max(0, (self.positioning_reward * (1 - distance_to_ball)))
            reward[rew_index] += positioning_bonus
            components["positioning_reward"][rew_index] += positioning_bonus
            
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
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
