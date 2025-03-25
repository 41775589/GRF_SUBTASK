import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling and sprinting through tight defensive lines."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For sticky actions including sprint and dribble

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_bonus": [0.0] * len(reward),
                      "sprint_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for sprinting; encouraging faster movement by checking sticky action for sprint.
            if 'sticky_actions' in o and o['sticky_actions'][8]:  # Index 8 corresponds to 'sprint'
                components["sprint_bonus"][rew_index] = 0.05
                reward[rew_index] += components["sprint_bonus"][rew_index]
                self.sticky_actions_counter[8] += 1

            # Reward for successful dribbling; ensuring agent uses dribble in congested areas.
            if 'sticky_actions' in o and o['sticky_actions'][9]:  # Index 9 corresponds to 'dribble'
                components["dribble_bonus"][rew_index] = 0.04
                reward[rew_index] += components["dribble_bonus"][rew_index]
                self.sticky_actions_counter[9] += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update sticky_actions usage count in info
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
