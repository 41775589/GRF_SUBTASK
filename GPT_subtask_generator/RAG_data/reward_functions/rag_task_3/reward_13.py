import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces a dense reward scheme focused on shooting skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_accuracy_reward = 1.0
        self.shot_power_reward = 1.0
        self.shot_pressure_reward = 0.5
        self.sticky_actions_mapping = {
            "action_sprint": 8,
            "action_dribble": 9,
            "action_shot": 12   # Assuming index 12 maps to a shot action in the action set
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        shot_reward = [0.0] * len(reward)
        components = {
            "base_score_reward": base_score_reward,
            "shot_accuracy_reward": [0.0] * len(reward),
            "shot_power_reward": [0.0] * len(reward),
            "shot_pressure_reward": [0.0] * len(reward)
        }

        for rew_index, _ in enumerate(reward):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1:  # Assuming 1 is the team of the agent
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance_to_goal = abs(ball_pos[0] - 1)  # goal at x=1 for simplicity
                proximity_to_defenders = min([np.linalg.norm(player_pos - opp_pos) for opp_pos in o['right_team']])
                
                # Reward for shot accuracy based on distance to the center of the goal
                if o['sticky_actions'][self.sticky_actions_mapping["action_shot"]] == 1:
                    shot_accuracy = max(0, 1 - distance_to_goal * 2)
                    components["shot_accuracy_reward"][rew_index] = shot_accuracy * self.shot_accuracy_reward
                    reward[rew_index] += components["shot_accuracy_reward"][rew_index]

                # Reward for shot power: assume directly related to sprinting speed
                if o['sticky_actions'][self.sticky_actions_mapping["action_sprint"]] == 1:
                    components["shot_power_reward"][rew_index] = self.shot_power_reward
                    reward[rew_index] += components["shot_power_reward"][rew_index]
                
                # Simulate pressure from defenders affecting the shot
                if proximity_to_defenders < 0.1:  # High pressure if a defender is within this range
                    components["shot_pressure_reward"][rew_index] = -self.shot_pressure_reward
                    reward[rew_index] += components["shot_pressure_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
