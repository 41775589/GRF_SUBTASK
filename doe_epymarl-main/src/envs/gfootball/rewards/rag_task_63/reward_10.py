import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for goalkeeper training tasks including shot stopping, quick distribution, and communication."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stopping_reward = 0.5
        self.ball_distribution_reward = 0.3
        self.communication_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_stopping_reward": [0.0] * len(reward),
                      "ball_distribution_reward": [0.0] * len(reward),
                      "communication_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['active'] == 0:  # Assuming index 0 is the goalkeeper
                # Analyzing ball stopping reward based on ball distance and save actions
                if o['ball_owned_team'] == 0 and o['sticky_actions'][2]:  # Active and 'action_top' used to stop
                    components["shot_stopping_reward"][rew_index] = self.shot_stopping_reward

                # Reward for appropriate ball distribution under pressure
                if o['game_mode'] in [3, 4, 6]:  # FreeKick, Corner or Penalty
                    if o['ball_owned_player'] == 0 and o['sticky_actions'][8]:  # Dribbling in pressure situations
                        components["ball_distribution_reward"][rew_index] = self.ball_distribution_reward

                # Communication rewards: placeholder for demonstration, advances with team play analysis
                teammates_positions = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
                if np.any(teammates_positions[:, 0] > o['ball'][0]):  # players are ahead in x-direction
                    components["communication_reward"][rew_index] = self.communication_reward

            # Calculate final reward for this observation
            reward[rew_index] += sum(components[c][rew_index] for c in components)

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
