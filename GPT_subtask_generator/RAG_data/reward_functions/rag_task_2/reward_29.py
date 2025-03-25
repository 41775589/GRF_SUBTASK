import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focusing on defensive teamwork and ball control transitions.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions = {}
        self._previous_ball_owner = None
        self._defensive_reward = 0.05
        self._lose_control_penalty = -0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions = {}
        self._previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._defensive_positions
        to_pickle['previous_ball_owner'] = self._previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_positions = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_owner = from_pickle.get('previous_ball_owner', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "lose_control_penalty": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_position = o['left_team'][o['active']]

            # Reward positioning between the ball and the goal if on the defensive team
            ball_pos = o['ball'][:2]
            own_goal_pos = [-1, 0]  # Assuming left team's goal for simplicity
            player_dist_to_goal = np.linalg.norm(active_player_position - own_goal_pos)
            ball_dist_to_goal = np.linalg.norm(ball_pos - own_goal_pos)

            if ball_dist_to_goal > player_dist_to_goal and o['ball_owned_team'] == 1:
                # More reward for being between the ball and own goal when opponent controls the ball
                components["defensive_reward"][rew_index] = self._defensive_reward
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Check if ball control is lost
            if (self._previous_ball_owner is not None and 
                self._previous_ball_owner == o['active'] and
                o['ball_owned_team'] != 0):
                components["lose_control_penalty"][rew_index] = self._lose_control_penalty
                reward[rew_index] += components["lose_control_penalty"][rew_index]

            # Update the last known ball owner if the current team owns it
            if o['ball_owned_team'] == 0:
                self._previous_ball_owner = o['ball_owned_player']

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
