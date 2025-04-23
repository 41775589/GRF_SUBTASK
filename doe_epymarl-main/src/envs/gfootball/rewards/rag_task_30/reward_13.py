import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense rewards focusing on defense to counterattack transitions and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define coefficients for the components of the reward
        self.defensive_positioning_reward = 0.5
        self.transition_reward = 0.3
        self.backward_movement_penalty = -0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No state specifics to load as of now
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Components that will be adjusted according to observations and actions
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward),
                      "transition_quality": [0.0] * len(reward),
                      "backward_movement_penalty": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Encourage strategic positioning when not possessing the ball
            if obs['ball_owned_team'] != 0:  # Check if our team does not own the ball
                x_position = obs['right_team'][obs['active']][0]  # x position of active agent
                if x_position < 0:  # Positioned in own half
                    components["defensive_positioning"][rew_index] = self.defensive_positioning_reward

            # Reward transitioning to counterattack
            if obs['ball_owned_team'] == 0 and obs['ball'][0] > 0:  # Ball is in the opponent's half
                components["transition_quality"][rew_index] = self.transition_reward

            # Penalize unnecessary backward movements
            for action, active in enumerate(obs['sticky_actions']):
                if action in [6, 7] and active == 1:  # Penalize the bottom and bottom left movements
                    components["backward_movement_penalty"][rew_index] = self.backward_movement_penalty

            # Calculate total reward for this agent
            reward[rew_index] = reward[rew_index] + \
                                components["defensive_positioning"][rew_index] + \
                                components["transition_quality"][rew_index] + \
                                components["backward_movement_penalty"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rewards, components = self.reward(reward)
        info["final_reward"] = sum(rewards)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, rewards, done, info
