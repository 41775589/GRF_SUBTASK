import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on specialized goalkeeper training with rewards for
    shot-stopping, quick reflexes, and initiating counter-attacks.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = 0  # Assuming the goalkeeper is always the first player in list
        self.shot_stopping_reward = 1.0
        self.reflex_reward = 0.5
        self.passing_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_stopping": [0.0] * len(reward),
                      "reflexes": [0.0] * len(reward),
                      "accurate_passing": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # For goalkeeper-specific training
            if o['active'] == self.goalkeeper_index:
                # Reward for shot stopping
                if o['game_mode'] == 6:  # Assuming Penalty mode involves shot stopping
                    components["shot_stopping"][rew_index] = self.shot_stopping_reward
                    reward[rew_index] += components["shot_stopping"][rew_index]

                # Reward for reflexes (based on ball movement magnitude and direction changes)
                ball_speed = np.linalg.norm(o['ball_direction'])
                if ball_speed > 0.05:
                    components["reflexes"][rew_index] = self.reflex_reward * ball_speed
                    reward[rew_index] += components["reflexes"][rew_index]
                
                # Reward for accurate passes (for counter-attack initiation)
                # Assuming that a successful 'long' kick or throw initiates a counter-attack
                if o['sticky_actions'][6] == 1 or o['sticky_actions'][5] == 1:
                    components["accurate_passing"][rew_index] = self.passing_reward
                    reward[rew_index] += components["accurate_passing"][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
