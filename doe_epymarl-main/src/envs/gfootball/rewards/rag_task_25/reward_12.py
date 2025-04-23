import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on dribbling and sprinting skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_and_sprint_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            dribbling = o['sticky_actions'][9]  # Check if dribbling in action
            sprinting = o['sticky_actions'][8]  # Check if sprinting
            ball_control = (o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'])

            if ball_control and dribbling and sprinting:
                components['dribble_and_sprint_reward'][i] = 0.05  # Reward for dribbling while sprinting

            # Sum up rewards for this step
            reward[i] = components['base_score_reward'][i] + components['dribble_and_sprint_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update counters for sticky actions
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
