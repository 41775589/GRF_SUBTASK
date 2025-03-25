import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper modifying score based on the behavior of a midfielder/advance defender."""

    def __init__(self, env):
        super().__init__(env)
        self._num_actions = 10  # number of total actions 
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)

        # Initialize specific action indices based on the environment's action space
        # Assuming indices corresponding to HighPass, LongPass, Sprint, and StopSprint actions in the original action set
        self.high_pass_index = 6
        self.long_pass_index = 7
        self.sprint_index = 8
        self.stop_sprint_index = 9
        self.dribble_index = 5
        
        # Reward magnitudes
        self.pass_reward = 0.05
        self.sprint_reward = 0.01
        self.dribble_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['wrapped_env_state'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['wrapped_env_state']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        additional_reward = [0.0] * len(reward)

        if observation is None:
            return reward, components
        
        for player_idx, obs in enumerate(observation):

            # Reward for successful high or long passes
            if obs['sticky_actions'][self.high_pass_index]:
                additional_reward[player_idx] += self.pass_reward
            if obs['sticky_actions'][self.long_pass_index]:
                additional_reward[player_idx] += self.pass_reward

            # Reward for sprinting to advantageous positions
            if obs['sticky_actions'][self.sprint_index]:
                additional_reward[player_idx] += self.sprint_reward
            if obs['sticky_actions'][self.stop_sprint_index]:
                additional_reward[player_idx] -= self.sprint_reward

            # Reward for maintaining control under pressure
            if obs['sticky_actions'][self.dribble_index]:
                additional_reward[player_idx] += self.dribble_reward

            reward[player_idx] += additional_reward[player_idx]

        components['additional_rewards'] = additional_reward
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
