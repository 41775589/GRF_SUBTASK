import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense high pass and crossing reward based on varying distances and angles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_pos = o['ball']
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_direction = o['ball_direction']
            game_mode = o['game_mode']

            # Calculate the distance and the directional vector from player to ball 
            distance = np.linalg.norm(np.array(player_pos) - np.array(ball_pos[:2]))
            direction = np.array(ball_pos[:2]) - np.array(player_pos)
            direction /= np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else 1

            # Check for high passes condition (distance > 0.2 and upward ball motion)
            if distance > 0.2 and ball_direction[2] > 0.02:
                # Reward for high pass
                components['high_pass_reward'][rew_index] = 0.3
                reward[rew_index] += components['high_pass_reward'][rew_index]

            # Check for crossing condition (large angle with goal line and crosses the field)
            goal_direction = np.array([1, 0]) if o['ball_owned_team'] == 0 else np.array([-1, 0])
            angle_with_goal = np.arccos(np.dot(direction, goal_direction) / (np.linalg.norm(direction) * np.linalg.norm(goal_direction)))
            if angle_with_goal > np.pi/3 and abs(ball_direction[1]) > 0.1:
                # Reward for crossing
                components['high_pass_reward'][rew_index] += 0.5
                reward[rew_index] += 0.5

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
