import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on offensive skills such as passing, shooting, and dribbling.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_reward = 0.1
        self.shot_reward = 0.3
        self.dribble_reward = 0.05
        self.sprint_reward = 0.02
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset sticky actions counter on environment reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Pass state saving logic to the environment's get_state. """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore state using the environment's set_state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Reward function emphasizing offensive skills (pass, shot, dribble, sprint).
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']

            # Adding pass rewards
            if sticky_actions[0] == 1 or sticky_actions[1] == 1:  # Assuming indices 0 and 1 relate to passing actions
                components['pass_reward'][rew_index] = self.pass_reward
                reward[rew_index] += components['pass_reward'][rew_index]

            # Adding shot rewards
            if sticky_actions[2] == 1:  # Assuming index 2 relates to the shot action
                components['shot_reward'][rew_index] = self.shot_reward
                reward[rew_index] += components['shot_reward'][rew_index]

            # Adding dribble rewards
            if sticky_actions[9] == 1:  # Assuming index 9 relates to dribbling
                components['dribble_reward'][rew_index] = self.dribble_reward
                reward[rew_index] += components['dribble_reward'][rew_index]

            # Adding sprint rewards
            if sticky_actions[8] == 1:  # Assuming index 8 relates to sprint
                components['sprint_reward'][rew_index] = self.sprint_reward
                reward[rew_index] += components['sprint_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
