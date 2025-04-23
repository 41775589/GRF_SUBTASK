import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that augments the football task with specific rewards to train a goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_efficiency = 0.1  # Reward contribution from goalkeeper specific actions

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
                      "goalkeeper_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for saving goals: negative reward if goal conceded
            if o['score'][0] < o['score'][1]:  # Assuming the keeper is on the left team
                components["goalkeeper_reward"][rew_index] = -1 * self.goalkeeper_efficiency
            else:
                # Rewards for ball blocking
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and o['left_team_roles'][o['active']] == 0:
                    components["goalkeeper_reward"][rew_index] = self.goalkeeper_efficiency
                    reward[rew_index] += 1 * components["goalkeeper_reward"][rew_index]

                # Penalty for bad positioning: If ball is near goal and not owned by keeper
                ball_pos = o['ball'][:2]  # X, Y coordinates of the ball
                goal_pos = [-1, 0]  # Assuming left team's goal X coordinate and center Y
                dist_to_goal = np.sqrt(np.sum((np.array(ball_pos) - np.array(goal_pos))**2))
                if dist_to_goal < 0.2 and o['ball_owned_team'] != 0:
                    components["goalkeeper_reward"][rew_index] = -self.goalkeeper_efficiency
                    reward[rew_index] += components["goalkeeper_reward"][rew_index]
                    
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
