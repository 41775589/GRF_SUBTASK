import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance defensive positions by adding rewards for quick stops and starts."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            d = np.linalg.norm(
                [o['left_team_direction'][o['active']],
                 o['right_team_direction'][o['active']]]
            )

            is_moving = d > 0.01
            last_action_stopped = self.sticky_actions_counter[o['active']] == 0 and not is_moving
            last_action_started = self.sticky_actions_counter[o['active']] == 0 and is_moving

            if last_action_stopped:
                components["transition_reward"][rew_index] = 0.5  # rewarding stopping
            elif last_action_started:
                components["transition_reward"][rew_index] = 0.5  # rewarding starting

            # Update reward with the transition components
            reward[rew_index] += components["transition_reward"][rew_index]

            # Update the action history
            if is_moving:
                self.sticky_actions_counter[o['active']] += 1
            else:
                self.sticky_actions_counter[o['active']] = 0

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
