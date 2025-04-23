import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specializes in goalkeeper coordination, focusing on strategies
    to back up the goalkeeper during high-pressure scenarios and to efficiently clear the ball
    to specific outfield players."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position_cache = None
        self.high_pressure_cache = {}
        self.outfield_pass_cache = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position_cache = None
        self.high_pressure_cache = {}
        self.outfield_pass_cache = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['goalkeeper_position_cache'] = self.goalkeeper_position_cache
        to_pickle['high_pressure_cache'] = self.high_pressure_cache
        to_pickle['outfield_pass_cache'] = self.outfield_pass_cache
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_position_cache = from_pickle['goalkeeper_position_cache']
        self.high_pressure_cache = from_pickle['high_pressure_cache']
        self.outfield_pass_cache = from_pickle['outfield_pass_cache']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_support_reward": [0.0] * len(reward),
                      "efficient_clearance_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']
            goalkeeper = o['left_team_roles'][o['active']] == 0  # Assuming 0 denotes goalkeeper

            # Rewarding goalkeeper positioning and action under high pressure
            if goalkeeper and ball_owned:
                d_ball_goal = np.linalg.norm(o['ball'] - [-1, 0])  # Distance of ball to own goal
                # Encourage the goalkeeper to clear the ball when close to goal under pressure
                if d_ball_goal < 0.3:
                    components["goalkeeper_support_reward"][rew_index] += 0.5
                    reward[rew_index] += components["goalkeeper_support_reward"][rew_index]

            # Reward for clearing the ball efficiently to outfield players
            if ball_owned and not goalkeeper:
                if self.goalkeeper_position_cache:
                    # Calculating distance to the last known goalkeeper position
                    d_to_gk = np.linalg.norm(o['ball'] - self.goalkeeper_position_cache)
                    # Reward for passing the ball to a safer area (away from the goalkeeper position)
                    if d_to_gk > 0.5:
                        components["efficient_clearance_reward"][rew_index] += 0.2
                        reward[rew_index] += components["efficient_clearance_reward"][rew_index]

            # Cache the goalkeeper's ball position when they last owned it
            if goalkeeper and ball_owned:
                self.goalkeeper_position_cache = o['ball'].copy()

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
