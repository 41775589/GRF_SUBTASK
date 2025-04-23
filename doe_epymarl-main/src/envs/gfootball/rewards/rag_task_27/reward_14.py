import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on defense."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.interception_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = to_pickle.get('state_info', {})
        state_info['interceptions'] = self.interceptions
        to_pickle['state_info'] = state_info
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle.get('state_info', {})
        self.interceptions = state_info.get('interceptions', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward)
        }
        
        # Check for interceptions.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] != -1:
                ball_owned_team = o['ball_owned_team']
                # Agent's team is the left team
                if ball_owned_team == 1 and 'right_team_active' in o:
                    # Only reward interceptions when ball was previously owned by the opponent
                    if o['left_team'][o['active']][0] < o['ball'][0]:  # player is ahead of the ball
                        intercept_dist = np.linalg.norm(o['ball'] - o['left_team'][o['active']])
                        # Reward the interception if very close to the ball when opponent team lost it.
                        if intercept_dist < 0.05:
                            components["interception_reward"][rew_index] += self.interception_reward
                            reward[rew_index] += components["interception_reward"][rew_index]
                            self.interceptions += 1
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
