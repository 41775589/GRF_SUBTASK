import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a coordinated offensive strategy reward to promote teamwork."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for rewarding passing and movement towards the goal
        self.pass_reward = 0.2
        self.goal_distance_reward = 0.1
        self.shoot_reward = 1  # Encourage shooting when near the goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No specific state to restore for this reward wrapper
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "goal_distance_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            distance_to_goal = np.linalg.norm([o['ball'][0] - 1, o['ball'][1]])  # distance to the opponent's goal

            # Reward for reducing the distance to the opponent's goal
            if distance_to_goal < 0.2:
                components["goal_distance_reward"][rew_index] = self.goal_distance_reward
                reward[rew_index] += components["goal_distance_reward"][rew_index]

            # Reward for passes made by the team
            if o['sticky_actions'][9]:  # indexing might depend on the action setup (assumed 9 is pass action)
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

            # Encourage shooting when in the close vicinity of the goal
            if distance_to_goal < 0.1 and o['sticky_actions'][6]:  # assume 6 is the shoot action
                components["shoot_reward"][rew_index] = self.shoot_reward
                reward[rew_index] += components["shoot_reward"][rew_index]

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
