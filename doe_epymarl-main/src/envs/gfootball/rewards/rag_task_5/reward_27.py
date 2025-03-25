import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive and counter-attack training reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Counter for the number of times the opponent advances in the player's defensive third
        self.opponent_in_defensive_third = 0
        # Counter for successful counter-attacks
        self.successful_counters = 0
        # Initialize the reward coefficients
        self.defensive_reward_coefficient = 0.05
        self.counter_attack_coefficient = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.opponent_in_defensive_third = 0
        self.successful_counters = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'opponent_in_defensive_third': self.opponent_in_defensive_third,
            'successful_counters': self.successful_counters
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.opponent_in_defensive_third = from_pickle['CheckpointRewardWrapper']['opponent_in_defensive_third']
        self.successful_counters = from_pickle['CheckpointRewardWrapper']['successful_counters']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Evaluate player's defensive actions
            if o['game_mode'] == 0 and o['ball_owned_team'] == 1:  # Ball owned by opponent
                ball_position = o['ball'][0]  # Using the x-coordinate
                if ball_position < -0.5:  # Ball in defensive third
                    self.opponent_in_defensive_third += 1
                    components["defensive_reward"][rew_index] += self.defensive_reward_coefficient
            
            # Evaluate counter-attack efficiency
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0:  # Ball owned by self
                ball_position = o['ball'][0]  # Using the x-coordinate
                if ball_position > 0.5:  # Successful progression to opponent's half
                    self.successful_counters += 1
                    components["counter_attack_reward"][rew_index] += self.counter_attack_coefficient
            
            reward[rew_index] += components["defensive_reward"][rew_index] + components["counter_attack_reward"][rew_index]

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
