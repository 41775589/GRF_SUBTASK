import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper encourages agents to specialize in initiating counterattacks 
    through long passes and quick transitions from defense to attack.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for additional reward components
        self.long_pass_reward_coeff = 0.2
        self.quick_transition_reward_coeff = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check if the ball was passed long towards the opposing half
            if o['ball_owned_team'] == o['active'] and np.linalg.norm(o['ball_direction'][:2]) > 0.2:
                components["long_pass_reward"][rew_index] += self.long_pass_reward_coeff

            # Reward quick transitions: if an agent's team recovers the ball and makes a quick move towards the opposing goal
            if o['ball_owned_team'] == o['active'] and o['game_mode'] in {2, 4, 5}:  # Consider game modes that involve regaining ball control
                components["quick_transition_reward"][rew_index] += self.quick_transition_reward_coeff

            # Update the reward for this agent
            reward[rew_index] += components["long_pass_reward"][rew_index] + components["quick_transition_reward"][rew_index]

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
