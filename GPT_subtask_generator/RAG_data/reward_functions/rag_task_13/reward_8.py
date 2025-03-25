import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to focus on defensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        components = {"base_score_reward": reward.copy()}
        modified_reward = reward.copy()

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] not in {0, 1}:  # Normal or KickOff mode
                continue

            if o['ball_owned_team'] == 0:
                ball_control_position = self.get_ball_control_position(o)
                if ball_control_position == 'defensive_third':
                    modified_reward[rew_index] += 0.05  # increase reward for having the ball in the defensive third
                    components['defensive_positioning'] = 0.05
                if o['designated'] == o['active'] and o['right_team_roles'][o['active']] in (1, 4):  # check if active player is a stopper or defensive role
                    modified_reward[rew_index] += 0.1
                    components['stopper_active'] = 0.1  # bonus for being actively controlling a stopper

            if not np.array_equal(o['ball'], o['left_team'][o['active']]):
                dist_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']])
                if dist_to_ball < 0.1:  # if close to ball while defending
                    modified_reward[rew_index] += 0.2
                    components['proximity_to_ball'] = 0.2  # incentive to stay close to the ball defensively

        return modified_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def get_ball_control_position(self, observation):
        x, y = observation['ball'][:2]
        if x > 0.5:
            return 'attacking_third'
        elif x > -0.5 and x <= 0.5:
            return 'midfield'
        else:
            return 'defensive_third'
