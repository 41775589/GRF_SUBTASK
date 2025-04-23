import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for offensive strategies including shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self._num_pass_checks = 1  # e.g., successful passing in crucial areas
        self._num_dribble_checks = 1  # e.g., successful dribbling past opponent players
        self._num_shot_checks = 1  # e.g., on target shots
        self.pass_reward = 0.2
        self.dribble_reward = 0.2
        self.shot_reward = 0.2
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
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Handling successful passes in critical regions
            if 'sticky_actions' in o and o['sticky_actions'][6] == 1 or o['sticky_actions'][9] == 1:  # action_bottom (for long/high pass)
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]
            
            # Handling dribbling through opponents
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # action_dribble
                # Check opponent's proximity and ball possession
                if any(np.linalg.norm(o['left_team'][o['active']] - p) < 0.05 for p in o['right_team']):
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]
            
            # Handling shooting and its accuracy
            if 'ball_direction' in o and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # own ball and current player
                ball_travel_direction = o['ball_direction']
                goal_direction = np.array([1, 0]) - o['ball'][:2]  # towards the right goal
                alignment = np.dot(ball_travel_direction[:2] / np.linalg.norm(ball_travel_direction[:2]), goal_direction / np.linalg.norm(goal_direction))
                if alignment > 0.7:  # roughly aligned towards the goal
                    components["shot_reward"][rew_index] = self.shot_reward
                    reward[rew_index] += components["shot_reward"][rew_index]

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
