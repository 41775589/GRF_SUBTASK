import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a checkpoint reward focused on defensive transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 10
        self._positions_collected = {'start_action': {}, 'stop_action': {}}
        self._position_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._positions_collected = {'start_action': {}, 'stop_action': {}}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._positions_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._positions_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "position_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_position = (o['right_team'][o['active']][0], o['right_team'][o['active']][1])
            active_action = o['sticky_actions']
            reward_increment = 0.0

            # Reward for starting motion from a stand-still
            if np.sum(active_action[:8]) > 0 and current_position not in self._positions_collected['start_action'].get(rew_index, set()):
                reward_increment += self._position_reward
                if rew_index in self._positions_collected['start_action']:
                    self._positions_collected['start_action'][rew_index].add(current_position)
                else:
                    self._positions_collected['start_action'][rew_index] = {current_position}

            # Reward for stopping motion
            if np.sum(active_action[:8]) == 0 and current_position not in self._positions_collected['stop_action'].get(rew_index, set()):
                reward_increment += self._position_reward
                if rew_index in self._positions_collected['stop_action']:
                    self._positions_collected['stop_action'][rew_index].add(current_position)
                else:
                    self._positions_collected['stop_action'][rew_index] = {current_position}
            
            reward[rew_index] += reward_increment
            components["position_reward"][rew_index] = reward_increment

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
