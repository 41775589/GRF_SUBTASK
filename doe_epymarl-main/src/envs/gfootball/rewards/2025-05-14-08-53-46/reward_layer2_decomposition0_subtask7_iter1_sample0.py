import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for defensive positioning to develop understanding of cutting off passes and spacing.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_bonus = 0.3  # Reward for positioning to cut off passing lanes.
        self.defensive_spacing_bonus = 0.2  # Reward for maintaining optimal spacing with other defenders.

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positional_reward": [0.0] * len(reward), "spacing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for agent_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # If the opponent has the ball
                # Calculate defensive positioning effectiveness
                opponent_ball_pos = o['ball'][:2]
                player_pos = o['left_team'][o['active']]
                distance_to_ball = np.linalg.norm(opponent_ball_pos - player_pos)
                if distance_to_ball < 0.2:  # Close to cutting off passes
                    components["positional_reward"][agent_index] = self.positional_bonus * (1 / (1 + distance_to_ball))
                    reward[agent_index] += components["positional_reward"][agent_index]

                # Calculate defensive spacing efficiency
                defensive_distances = []
                for position in o['left_team']:
                    if not np.array_equal(position, player_pos):
                        defensive_distances.append(np.linalg.norm(player_pos - position))
                
                if defensive_distances:
                    mean_def_dist = np.mean(defensive_distances)
                    if 0.1 < mean_def_dist < 0.3:  # Optimal defensive spacing reward
                        components["spacing_reward"][agent_index] = self.defensive_spacing_bonus
                        reward[agent_index] += components["spacing_reward"][agent_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Summing reward for info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_active
        return observation, reward, done, info
