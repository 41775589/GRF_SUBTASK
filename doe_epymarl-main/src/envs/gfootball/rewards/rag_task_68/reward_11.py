import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on training offensive strategies in the Google Research Football
    environment by providing incentives for mastering accurate shooting, dribbling, and different
    types of passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset the sticky actions counter when the environment is reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Saves the sticky actions state. """
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restores the sticky actions state. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function to promote offensive strategies: accurate shooting,
        effective dribbling and using varied passes (long/high passes).
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned = o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']

            # Reward for dribbling - Check if dribbling action is active
            if ball_owned and o['sticky_actions'][9] == 1:  # 'action_dribble' is at index 9
                components["dribbling_reward"][rew_index] = 0.02
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Reward for successful shooting
            if ball_owned and np.linalg.norm(o['ball_direction'][:2]) > 0.05:
                # Assume a higher ball speed in direction as a proxy for shooting towards the goal
                components["shooting_reward"][rew_index] = 0.1
                reward[rew_index] += components["shooting_reward"][rew_index]
                
            # Reward for using long/high passes 
            if ball_owned and o['ball'][2] > 0.1:  # Simple assumption: if z component of ball position is high
                components["passing_reward"][rew_index] = 0.03
                reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Step through the environment, modifying reward and providing additional info in outputs. """
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
