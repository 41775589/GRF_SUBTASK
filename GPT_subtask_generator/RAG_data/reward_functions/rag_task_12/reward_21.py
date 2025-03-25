import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on actions relevant to the agent described in the task."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward magnitudes for high pass, long pass, dribble, sprint actions
        self.actions_reward_mapping = {
            'high_pass': 0.05,
            'long_pass': 0.05,
            'sprint': 0.02,
            'dribble': 0.03,
            'stop_sprint': 0.01
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Initialize any stored state if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Check and reward relevant actions
        for i, o in enumerate(observation):
            action_ids = {
                8: 'sprint',
                9: 'dribble',
                # Assuming action IDs for high_pass and long_pass
                'high_pass': 10,  # Hypothetical ID
                'long_pass': 11   # Hypothetical ID
            }
            sticky_actions = o.get('sticky_actions', [])

            for action_index, is_action_active in enumerate(sticky_actions):
                if is_action_active > 0 and action_index in action_ids:
                    action_name = action_ids[action_index]
                    if action_name in self.actions_reward_mapping:
                        components['action_rewards'][i] += self.actions_reward_mapping[action_name]

            # Check for 'stop_sprint' which is the stopping of action 'sprint'
            if sticky_actions[8] == 0 and self.sticky_actions_counter[8] > 0:
                components['action_rewards'][i] += self.actions_reward_mapping['stop_sprint']

            reward[i] += components['action_rewards'][i]

        # Update the sticky_actions_counter for the next step
        self.sticky_actions_counter = np.array([o['sticky_actions'] for o in observation]).sum(axis=0)

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
