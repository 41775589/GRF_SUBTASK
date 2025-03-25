import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defensive responsiveness and interception skill-oriented reward.
    It rewards the agent for successful interceptions and maintaining a strategic defensive position.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize counters and thresholds
        self.interception_bonus = 1.0
        self.positioning_reward = 0.01
        self.close_to_opponent_bonus = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save additional state information if necessary
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Get current observations from the environment
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for intercepting the ball
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'designated' in o and o['active'] == o['designated']:
                components["interception_reward"][rew_index] = self.interception_bonus
                reward[rew_index] = components["base_score_reward"][rew_index] + components["interception_reward"][rew_index]

            # Reward for being strategically well-positioned defensively (close to an opponent who controls the ball)
            if o['ball_owned_team'] == 0:  # Assumption: our team's team index is 0
                min_dist = np.min(np.linalg.norm(o['left_team'] - o['ball'], axis=1))
                if min_dist < 0.1:  # A threshold distance which considers being close enough to intercept
                    components["positioning_reward"][rew_index] = self.close_to_opponent_bonus - min_dist * 10 * self.positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Execute step in the environment
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = int(action)
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
