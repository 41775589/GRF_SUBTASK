import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward specifically designed to promote
    effective short passes under defensive pressure for ball retention and effective distribution.
    This encourages agents to optimize for maintaining possession of the ball and making strategic
    plays under pressure, aiding in defensive stability and counter-attack transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_success_counter = 0
        self.possession_reward_multiplier = 0.1
        self.pass_reward_multiplier = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_success_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_success_counter'] = self.pass_success_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_success_counter = from_pickle['pass_success_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for maintaining possession under pressure
            if o['ball_owned_team'] == 0:  # Assuming agent's team is team 0
                # Base possession reward
                components["possession_reward"][rew_index] = self.possession_reward_multiplier
                reward[rew_index] += components["possession_reward"][rew_index]

                # Reward successful passes
                if 'action' in o and o['action'] == 'short_pass' and o['game_mode'] == 0:
                    self.pass_success_counter += 1
                    components["passing_reward"][rew_index] = self.pass_reward_multiplier * self.pass_success_counter
                    reward[rew_index] += components["passing_reward"][rew_index]

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
