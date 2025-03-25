import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focusing on mastering defensive responsiveness and 
    interception skills, rewarding based on positional adjustments, interceptions, 
    and maintaining defensive line integrity under high-pressure scenarios.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initializing counters or trackers
        self.interception_reward = 0.3
        self.positional_reward = 0.1
        self.pressure_reward = 0.15

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
                      "interception_reward": [0.0] * len(reward),
                      "positional_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]

            # Rewarding for ball interception
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                # Previous possession wasn't by the current team
                if self.previous_ball_owned_team == 1:
                    reward[i] += self.interception_reward
                    components["interception_reward"][i] = self.interception_reward

            # Reward for maintaining optimal position relative to the ball and goals
            player_pos = obs['left_team'][obs['active']][:2]
            ball_pos = obs['ball'][:2]
            goal_pos = np.array([-1, 0])  # Simulating left team defensive goal

            # Calculate distances
            dist_to_ball = np.linalg.norm(player_pos - ball_pos)
            dist_to_goal = np.linalg.norm(player_pos - goal_pos)

            # Encourage being between the ball and own goal (defensive positioning)
            if dist_to_ball < dist_to_goal:
                reward[i] += self.positional_reward
                components["positional_reward"][i] = self.positional_reward

            # Reward for handling pressure (multiple opponents nearby)
            opponent_positions = obs['right_team']
            close_opponents = np.sum(np.linalg.norm(opponent_positions - player_pos, axis=1) < 0.2)
            if close_opponents >= 2:  # Handling multiple opponents
                reward[i] += self.pressure_reward
                components["pressure_reward"][i] = self.pressure_reward

            # Keep the state of the last possession for next reward calculations
            self.previous_ball_owned_team = obs['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
