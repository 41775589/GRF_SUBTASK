import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for mastering accurate shooting, effective dribbling,
    and using advanced passing strategies.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else sum(env.action_space.nvec)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward),
                      "shooting_reward": np.zeros(4),
                      "dribbling_reward": np.zeros(4),
                      "passing_reward": np.zeros(4)}
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]

            # Reward shooting accuracy
            if obs['ball_direction'][2] > 0.5:  # Assuming z direction signifies upward shot
                components['shooting_reward'][i] += 0.2

            # Reward maintaining possession while dribbling
            if 9 in obs['sticky_actions']:  # action_dribble
                components['dribbling_reward'][i] += 0.1 * np.linalg.norm(obs['ball_direction'][:2])

            # Reward effective passing
            if obs['game_mode'] in {3, 5}:  # FreeKick or ThrowIn as opportunities after long/high passes
                components['passing_reward'][i] += 0.3

            # Add components to the reward
            reward[i] += components['shooting_reward'][i] + components['dribbling_reward'][i] + components['passing_reward'][i]

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
