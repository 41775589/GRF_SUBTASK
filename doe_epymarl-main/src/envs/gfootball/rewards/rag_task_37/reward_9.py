import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that encourages advanced ball control and passing under pressure.
    The reward is enhanced when executing perfect passes under tight game situations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_reward = 0.05

    def reset(self):
        """ Resets the environment and sticky actions counter when a new game starts. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Saves the current state of the wrapper along with the environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the state of the wrapper along with the environment's state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward function to emphasize skilled passes when under defensive pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                # Consider the pressure situation by proximity of opponent players
                min_distance_to_opponent = np.min(np.linalg.norm(o['left_team'] - o['right_team'][o['active']], axis=1))
                pressure_threshold = 0.05  # Assuming pressured if an opponent is very close

                # Check if the sticky action indicates a pass (Short Pass, High Pass, Long Pass)
                if o['sticky_actions'][6] or o['sticky_actions'][7] or o['sticky_actions'][8]:
                    if min_distance_to_opponent < pressure_threshold:
                        components["passing_reward"][rew_index] = self._pass_reward * 2  # More reward under pressure
                    else:
                        components["passing_reward"][rew_index] = self._pass_reward

                reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Step function to apply the reward function adjustments
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
