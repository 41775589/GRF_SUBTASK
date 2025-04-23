import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A Gym reward wrapper that encourages offensive plays including shooting,
    advanced dribbling, and a variety of pass types.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shoot_reward_coeff = 0.5
        self.dribble_reward_coeff = 0.3
        self.pass_reward_coeff = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:  # If the controlled team owns the ball
                # Shooting attempt - we assume this when the team tries a shot towards the goal
                if obs['game_mode'] == 6:  # Penalty mode considered as a direct attempt to score
                    components["shoot_reward"][rew_index] = self.shoot_reward_coeff
                    reward[rew_index] += components["shoot_reward"][rew_index]

                # Dribbling, inferred from multiple sticky action occurrences 
                # of action_dribble at indices 8 and 9 (dribbling actively)
                if obs['sticky_actions'][9] == 1:  # dribble active
                    components["dribble_reward"][rew_index] = self.dribble_reward_coeff
                    reward[rew_index] += components["dribble_reward"][rew_index]

                # Passing rewards, here focusing on longer and high-effort passes
                if 'ball_direction' in obs and np.linalg.norm(obs['ball_direction'][:2]) > 0.1:  # arbitrary threshold for a "long/high" pass
                    components["pass_reward"][rew_index] = self.pass_reward_coeff
                    reward[rew_index] += components["pass_reward"][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
