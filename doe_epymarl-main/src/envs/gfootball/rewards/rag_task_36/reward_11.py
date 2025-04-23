import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on dribbling maneuvers and player positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = None
        self.dribble_state = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = None
        self.dribble_state = False
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {
            "base_score_reward": reward,
            "dribbling_reward": 0.0,
            "positioning_reward": 0.0
        }
        
        ball_owned_team = observation[0]['ball_owned_team']
        dribbling_action = observation[0]['sticky_actions'][9]  # action_dribble index is 9
        active_player_pos = observation[0]['right_team'][observation[0]['active']] \
            if observation[0]['ball_owned_team'] == 1 else observation[0]['left_team'][observation[0]['active']]

        # Reward for starting and stopping dribble
        if ball_owned_team == 1 and dribbling_action and not self.dribble_state:
            components['dribbling_reward'] += 0.05
            self.dribble_state = True
        elif ball_owned_team == 1 and not dribbling_action and self.dribble_state:
            components['dribbling_reward'] += 0.05
            self.dribble_state = False
        
        # Reward for good positioning (fluid transitions between defense and offense)
        if ball_owned_team == 1:  # Team 1 is the right team, modify accordingly if needed
            opponent_goal = np.array([1, 0])  # Opponent goal position at the right
            distance_to_goal = np.linalg.norm(active_player_pos - opponent_goal)
            components['positioning_reward'] += max(0, (0.05 - distance_to_goal * 0.05))
        
        # Apply components to reward
        reward += components['dribbling_reward'] + components['positioning_reward']
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
