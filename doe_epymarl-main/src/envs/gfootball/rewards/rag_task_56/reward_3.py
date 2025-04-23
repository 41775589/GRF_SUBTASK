import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive play improvements, tailored for goalkeepers and defenders."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.defender_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.defender_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['goalkeeper_rewards'] = self.goalkeeper_rewards
        to_pickle['defender_rewards'] = self.defender_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_rewards = from_pickle.get('goalkeeper_rewards', {})
        self.defender_rewards = from_pickle.get('defender_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_play_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Goalkeeper's shot-stopping rewards
            if o['active'] == 0 and o['left_team_roles'][o['active']] == 0:  # Assuming 0 is the goalkeeper
                if self.goalkeeper_rewards.get(rew_index, 0) == 0 and o['ball_owned_team'] == 0:
                    components["defensive_play_reward"][rew_index] += 0.5  # reward goalkeeper for having the ball
                    self.goalkeeper_rewards[rew_index] = 1

            # Defenders tackling and retention bonus
            if o['left_team_roles'][o['active']] in [1, 2, 3]:  # Assuming these indices correspond to defenders
                if self.defender_rewards.get(rew_index, 0) == 0 and o['ball_owned_team'] == 0:
                    components["defensive_play_reward"][rew_index] += 0.3  # reward defenders for having the ball
                    self.defender_rewards[rew_index] = 1

            # Update the reward to include the defensive components
            reward[rew_index] += components["defensive_play_reward"][rew_index]

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
