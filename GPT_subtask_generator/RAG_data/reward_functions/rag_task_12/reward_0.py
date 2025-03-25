import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on passing, dribbling, and dynamic movement."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passing_bonus = 0.1
        self._dribbling_bonus = 0.1
        self._movement_bonus = 0.05
        self._previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        to_pickle['prev_ball_pos'] = self._previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = from_pickle['prev_ball_pos']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(),
                      "dynamic_movement_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward),
                      "dribbling_bonus": [0.0] * len(reward)}
        
        ball_position = observation[0]['ball']
        ball_owned_team = observation[0]['ball_owned_team']

        for i, obs in enumerate(observation):
            if ball_owned_team == obs['ball_owned_team'] and ball_owned_team != -1:
                if 'action_high_pass' in obs['sticky_actions'] or 'action_long_pass' in obs['sticky_actions']:
                    components["passing_bonus"][i] = self._passing_bonus
                if 'action_dribble' in obs['sticky_actions']:
                    components["dribbling_bonus"][i] = self._dribbling_bonus
                if self._previous_ball_position is not None:
                    movement_distance = np.linalg.norm(np.array(ball_position) - np.array(self._previous_ball_position))
                    if movement_distance > 0.01:
                        components["dynamic_movement_bonus"][i] = self._movement_bonus
        
        self._previous_ball_position = ball_position

        # Sum all components to form final reward
        for i in range(len(reward)):
            reward[i] += components["passing_bonus"][i] + components["dribbling_bonus"][i] + components["dynamic_movement_bonus"][i]
        
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
