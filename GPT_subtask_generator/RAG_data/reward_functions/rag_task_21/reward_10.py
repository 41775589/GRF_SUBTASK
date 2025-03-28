import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._interception_rewards = {}
        self.interception_reward = 0.8
        self.ball_progression_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._interception_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._interception_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._interception_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for ball interceptions in the defense area
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['ball'][0] < 0:
                if self._interception_rewards.get(rew_index, 0) == 0:
                    components["interception_reward"][rew_index] = self.interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]
                    self._interception_rewards[rew_index] = 1

            # Reward if the player is in position during opponent's possession
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball']
                dist_to_ball = np.linalg.norm(player_pos[:2] - ball_pos[:2])
                # Reward defenders being close to the ball when they don't have the ball
                if dist_to_ball < 0.2:
                    reward[rew_index] += self.ball_progression_reward * (0.2 - dist_to_ball)

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
