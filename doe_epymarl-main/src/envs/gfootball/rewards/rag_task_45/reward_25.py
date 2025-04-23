import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on abruptly starting and stopping actions for defensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_action = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_action = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_action': self.previous_action
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_action = from_pickle['CheckpointRewardWrapper']['previous_action']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_start_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_action = np.argmax(o['sticky_actions'])  # assumes one-hot encoded action

            # Reward players for rapidly changing to/from sprint/dribble or stopping
            if self.previous_action is not None:
                if self.previous_action in [8, 9] and current_action not in [8, 9]:
                    # Reward for stopping sprint/dribble
                    components["stop_start_reward"][rew_index] = 0.5
                elif self.previous_action not in [8, 9] and current_action in [8, 9]:
                    # Reward for starting sprint/dribble
                    components["stop_start_reward"][rew_index] = 0.5

            reward[rew_index] += components["stop_start_reward"][rew_index]
            self.previous_action = current_action

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
