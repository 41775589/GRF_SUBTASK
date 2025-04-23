import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward based on effective mid to long-range passing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward_multiplier = 0.2
        self.team_efficiency_reward_multiplier = 0.1

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
                      "pass_quality_reward": [0.0] * len(reward),
                      "team_efficiency_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            previous_ball_position = np.array(o['ball'])
            
            # Reward for effective passing based on ball control and movement
            if (o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']):
                new_ball_position = np.array(o['ball'])
                ball_moved_distance = np.linalg.norm(new_ball_position - previous_ball_position)
                
                # Mid to long-range passes are those above a certain threshold distance
                if ball_moved_distance > 0.3:  # Threshold for considering it a significant pass
                    components["pass_quality_reward"][rew_index] = self.pass_quality_reward_multiplier * ball_moved_distance
                    reward[rew_index] += components["pass_quality_reward"][rew_index]

            # Team play efficiency: encourages maintaining ball control after passing
            current_player_position = np.array(o['left_team'][o['active']])
            closest_teammate = np.inf
            
            for teammate_idx, teammate_pos in enumerate(o['left_team']):
                if teammate_idx != o['active']:  # Exclude self
                    distance_to_teammate = np.linalg.norm(current_player_position - np.array(teammate_pos))
                    if distance_to_teammate < closest_teammate:
                        closest_teammate = distance_to_teammate
            
            # Reward for keeping the ball within close proximity to teammates
            if closest_teammate < 0.15:  # Teammates are close after a pass
                components["team_efficiency_reward"][rew_index] = self.team_efficiency_reward_multiplier
                reward[rew_index] += components["team_efficiency_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
