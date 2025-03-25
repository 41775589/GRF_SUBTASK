import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.1
        self.shooting_window_reward = 0.15
        self.position_advantage_reward = 0.05

    def reset(self, **kwargs):
        # Reset and return initial observation
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Get environment's observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),  # basic game-based reward
                      "strategic_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "shooting_window_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owner_team = o['ball_owned_team']

            # Reward for strategic passing plays
            if ball_owner_team == 1 and 'ball_owned_player' in o:
                owner = o['ball_owned_player']
                # Check if there's forward movement towards the opponent's goal, passing forwards
                if np.cos(o['ball_direction'][0]) > 0:
                    # Assuming the angle to the opponent's goal is narrower
                    components['strategic_pass_reward'][rew_index] = self.passing_reward
                    reward[rew_index] += components['strategic_pass_reward'][rew_index]

            # Reward for positioning to receive passes or be ready to shoot
            if ball_owner_team == 1:
                goal_distance = np.linalg.norm(o['ball'][:2])  # only x, y
                if goal_distance < 0.3:  # closer to the goal
                    components['positioning_reward'][rew_index] = self.position_advantage_reward
                    reward[rew_index] += components['positioning_reward'][rew_index]

            # Reward for being in a good shooting window
            if 'designated' in o and o['designated'] == o['active']:
                x, y = o['ball'][:2]
                # Check if the player is situated in 'key' shooting areas
                if x > 0.7 and abs(y) < 0.2:
                    components['shooting_window_reward'][rew_index] = self.shooting_window_reward
                    reward[rew_index] += components['shooting_window_reward'][rew_index]

        return reward, components

    def step(self, action):
        # Perform a step in the environment
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
