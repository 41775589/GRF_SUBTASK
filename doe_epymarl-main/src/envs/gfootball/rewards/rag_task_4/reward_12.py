import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward for dribbling skills and use of sprint in offensive moves.
    It provides additional rewards for controlling the ball, moving closer to the goal, 
    and effectively using sprint actions in tight defensive setups.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_control_reward = 0.1
        self._sprint_bonus = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control_reward": [0.0] * len(reward),
            "sprint_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward the player for having control of the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                components["ball_control_reward"][rew_index] = self._ball_control_reward
                reward[rew_index] += components["ball_control_reward"][rew_index]

            # If 'Sprint' action is active, give a bonus if moving forward with the ball
            if 'sticky_actions' in o and o['sticky_actions'][8] > 0:
                components["sprint_bonus"][rew_index] = self._sprint_bonus
                reward[rew_index] += components["sprint_bonus"][rew_index]

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
