import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on quick counter-attacks after gaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_ownership = None
        self.counter_attack_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_ownership = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_ownership'] = self.previous_ball_ownership
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_ownership = from_pickle['previous_ball_ownership']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward logic to encourage quick decision-making and efficient ball
        handling for initiating counter-attacks.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'counter_attack_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_ownership = o['ball_owned_team']

            # Check if the team just gained possession of the ball
            if (current_ball_ownership == 0) and (self.previous_ball_ownership == 1):
                # Reward for quick control and playmaking after gaining possession
                components['counter_attack_reward'][rew_index] = self.counter_attack_reward
                reward[rew_index] += components['counter_attack_reward'][rew_index]

            self.previous_ball_ownership = current_ball_ownership

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
