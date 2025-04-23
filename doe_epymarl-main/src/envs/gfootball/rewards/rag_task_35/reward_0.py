import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on strategic player positioning and movement optimization."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_in_control_last_position = None
        self.distances_from_goal = []
        self.last_score = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_in_control_last_position = None
        self.distances_from_goal = []
        self.last_score = self.env.unwrapped.observation()['score']
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['initial_player_position'] = self.ball_in_control_last_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_in_control_last_position = from_pickle['initial_player_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positional_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        current_score = observation['score']
        for team_idx in range(len(reward)):
            score_diff = current_score[team_idx] - self.last_score[team_idx]
            if score_diff > 0:
                components["positional_reward"][team_idx] = 1.0
            
            ball_pos = observation['ball'][:2]
            player_idx = observation['ball_owned_team']
            if player_idx == team_idx and player_idx != -1:
                if self.ball_in_control_last_position is None:
                    self.ball_in_control_last_position = ball_pos
                
                current_dist_from_goal = np.linalg.norm(ball_pos - np.array([1, 0]))
                last_dist_from_goal = np.linalg.norm(self.ball_in_control_last_position - np.array([1, 0]))

                distance_improvement = last_dist_from_goal - current_dist_from_goal
                components["positional_reward"][team_idx] += distance_improvement * 0.1
                self.ball_in_control_last_position = ball_pos

            reward[team_idx] += sum([x for x in components.values()])[team_idx]

        self.last_score = current_score
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
