import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that incentivizes agents for successful high passes and cross plays.
    Rewards are given based on the crossing distance, height of the ball, and successful
    reception by another team member, enhancing play strategies and spatial awareness.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_crossing_distance = 0.2  # Minimum required distance for a crossing to be rewarded
        self.min_height = 0.15  # Minimum height for a pass to be considered a high pass
        self.pass_reward = 0.3  # Reward for a successful high pass
        self.cross_success_reward = 0.5  # Additional reward for a successful cross, e.g., leading to a shot
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "cross_success_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            # Check heights and distance for passes
            ball_z = o['ball'][2]
            if ball_z >= self.min_height:
                player_positions = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                ball_pos = np.array(o['ball'][:2])
                player_pos = player_positions[o['ball_owned_player']]

                # check for change in possession that maintains team control for a minimum distance
                next_player = self.find_nearest_teammate(o['ball_owned_team'], ball_pos, player_positions, o['ball_owned_player'])
                if next_player is not None:
                    dist = np.linalg.norm(player_pos - player_positions[next_player])
                    if dist >= self.min_crossing_distance:
                        components["pass_reward"][i] = self.pass_reward
                        if o['right_team_roles'][next_player] == 9:  # Assuming role 9 implies attacking roles
                            components["cross_success_reward"][i] = self.cross_success_reward

            reward[i] += components["pass_reward"][i] + components["cross_success_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def find_nearest_teammate(self, team_id, ball_position, player_positions, current_player_index):
        min_dist = float('inf')
        nearest_player = None
        for idx, position in enumerate(player_positions):
            if idx != current_player_index:
                dist = np.linalg.norm(position - ball_position)
                if dist < min_dist:
                    nearest_player = idx
                    min_dist = dist
        return nearest_player
