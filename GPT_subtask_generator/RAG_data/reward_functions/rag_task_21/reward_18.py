import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive-focused checkpoint reward aimed to improve defensive responsiveness and interception skills."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize tracking for stateful rewards
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "interception_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the opponent has the ball and active player is close to the ball.
            if o['ball_owned_team'] == 1 and o['left_team'][o['active']][0] > o['ball'][0]:
                distance_to_ball = np.linalg.norm(o['left_team'][o['active']][:2] - o['ball'][:2])
                if distance_to_ball < 0.1:  # close enough to attempt a tackle
                    components["interception_reward"][rew_index] = 0.1  # small reward for being close to the ball
                if distance_to_ball < 0.05:  # very close to the ball
                    components["interception_reward"][rew_index] = 1.0  # higher reward for potential interception

        # Sum the components for each agent's reward
        final_rewards = [sum(x) for x in zip(reward, components["interception_reward"])]

        return final_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        original_rewards = reward.copy()
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track and report sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
