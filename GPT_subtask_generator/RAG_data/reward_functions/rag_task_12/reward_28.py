import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for a midfielder/defender agent."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, rew in enumerate(reward):
            obs = observation[index]

            # Custom reward for successful passes
            if obs['game_mode'] in [2, 3]:  # Assuming modes 2 and 3 relate to pass plays
                if obs['ball_owned_team'] == obs['active']:  # Ball owned by the agent's team
                    components["passing_reward"][index] = 0.2
                    reward[index] += components["passing_reward"][index]

            # Reward for dribbling under pressure
            if obs['sticky_actions'][9]:  # Assuming 9 corresponds to dribbling action
                defenders_nearby = np.any(
                    np.linalg.norm(obs['left_team'] - obs['ball'][:2], axis=1) < 0.1)
                if defenders_nearby:
                    components["dribbling_reward"][index] = 0.3
                    reward[index] += components["dribbling_reward"][index]

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
