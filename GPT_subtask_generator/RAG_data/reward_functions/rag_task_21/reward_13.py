import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focusing on defensive
    maneuvers, intercepting the ball, and maintaining effective positions relative
    to the ball and opponent players.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_reward = 0.2
        self.positioning_reward = 0.1
        self.ball_intercepted = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_intercepted = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_intercepted'] = self.ball_intercepted
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_intercepted = from_pickle['ball_intercepted']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:
                # When opponent team owns the ball, check for interception opportunities
                if self.ball_intercepted.get(rew_index, False):
                    components["intercept_reward"][rew_index] = self.intercept_reward
                if o.get('active', -1) == o.get('ball_owned_player', -2):
                    self.ball_intercepted[rew_index] = True
                    reward[rew_index] += self.intercept_reward * 1.5

            # Evaluate the defensive positioning of the agent relative to the ball and the active player
            ball_pos = o['ball'][:2]
            agent_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 1 else o['right_team'][o['active']]
            distance_to_ball = np.sqrt(np.sum(np.square(np.array(ball_pos) - np.array(agent_pos))))

            # Add rewards for maintaining good defensive positions
            if distance_to_ball < 0.2:  # good proximity to intervene or intercept
                components["positioning_reward"][rew_index] = self.positioning_reward
                reward[rew_index] += self.positioning_reward * 1.2
        
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
