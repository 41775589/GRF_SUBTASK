import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding offensive strategies like shooting, dribbling, and passing in the Google Research Football environment."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._shooting_reward = 1.0
        self._dribbling_reward = 0.5
        self._passing_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Assuming the fourth index in `sticky_actions` array corresponds to shooting
        # Eighth and ninth correspond to dribbling (sprint and dribble actions)
        # Assuming accurate shots, effective dribbling and strategic passing are monitored
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['sticky_actions'][4]:  # if shooting action is active
                if o['ball'][0] > 0:  # assuming positive x-direction is towards opponent's goal
                    components["shooting_reward"][rew_index] = self._shooting_reward
            if o['sticky_actions'][8] or o['sticky_actions'][9]:  # if dribbling or sprinting
                components["dribbling_reward"][rew_index] = self._dribbling_reward
            if o['game_mode'] in [2, 5]:  # Game modes indicating passing (goal kick, throw-in possibly indicating long passes)
                components["passing_reward"][rew_index] = self._passing_reward

            reward[rew_index] += (components["shooting_reward"][rew_index] +
                                  components["dribbling_reward"][rew_index] +
                                  components["passing_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update action counters
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
