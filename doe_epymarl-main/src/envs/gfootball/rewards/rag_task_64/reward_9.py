import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successfully executing high passes and crossing
    from varying distances and angles to enhance dynamic attacking plays.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_weights = np.linspace(0.1, 1.0, num=10)  # Linear weights for pass quality

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
                      "high_pass_crossing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball'][:2]  # Focus on x, y position
            ball_direction = o['ball_direction'][:2]
            ball_owned_team = o['ball_owned_team']
            active_player_pos = (o['left_team'] if ball_owned_team == 0 else o['right_team'])[o['active']]

            # Calculate directive vector from player to the goal
            goal_pos = [1, 0] if ball_owned_team == 0 else [-1, 0]
            direction_to_goal = np.subtract(goal_pos, active_player_pos)

            # Normalization
            norm_ball_direction = np.linalg.norm(ball_direction)
            norm_direction_to_goal = np.linalg.norm(direction_to_goal)

            if norm_ball_direction != 0 and norm_direction_to_goal != 0:
                # Calculate the angle between the pass direction and the direction towards opponent's goal
                cosine_angle = np.dot(ball_direction, direction_to_goal) / (norm_ball_direction * norm_direction_to_goal)
                # High passes are often upwards, considering the z dimension of the ball's movement
                is_high_pass = o['ball'][2] > 0.1  # Arbitrary threshold for z > 0.1 as 'high'

                if cosine_angle > 0.8 and is_high_pass:  # Cosine of angle threshold for 'forward direction'
                    checkpoint_index = int((cosine_angle - 0.8) / 0.02)  # Map cosine range to indices
                    checkpoint_index = min(checkpoint_index, len(self.pass_quality_weights) - 1)
                    components["high_pass_crossing_reward"][rew_index] = self.pass_quality_weights[checkpoint_index]
                    reward[rew_index] += components["high_pass_crossing_reward"][rew_index]

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
