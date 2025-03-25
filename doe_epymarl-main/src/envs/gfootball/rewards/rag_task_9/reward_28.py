import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on offensive skills such as passing, shooting, and dribbling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for different offensive actions.
        self.pass_reward = 0.05
        self.shoot_reward = 0.1
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = 'state_info_here'
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Update wrapper-specific state if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for i in range(len(reward)):
            o = observation[i]
            components['pass_reward'] = 0.0
            components['shoot_reward'] = 0.0
            components['dribble_reward'] = 0.0
            components['sprint_reward'] = 0.0

            # Check the sticky actions for corresponding rewards.
            if o['sticky_actions'][9] == 1:  # sprint
                components['sprint_reward'] += self.sprint_reward
            if o['sticky_actions'][8] == 1:  # dribble
                components['dribble_reward'] += self.dribble_reward
            if o['sticky_actions'][1] == 1 or o['sticky_actions'][2] == 1:  # short or long pass
                components['pass_reward'] += self.pass_reward
            if o['game_mode'] == 6 and o['active'] == o['ball_owned_player']:  # shot (mode penalty)
                components['shoot_reward'] += self.shoot_reward

            # Sum up all component rewards
            total_additional_reward = sum(components.values())
            reward[i] += total_additional_reward

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
