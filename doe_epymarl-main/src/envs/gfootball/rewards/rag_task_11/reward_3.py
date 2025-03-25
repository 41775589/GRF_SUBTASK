import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a precision-based finishing and fast-paced maneuvers reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        components = {"base_score_reward": reward.copy(),
                      "precision_reward": [0.0] * len(reward),
                      "pace_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Add precision-based finishing reward
            if o['ball_owned_team'] == 1:  # assuming agent's team is 1
                goal_distance = np.sqrt(o['ball'][0]**2 + o['ball'][1]**2)
                if goal_distance < 0.1:  # close to the goal
                    components['precision_reward'][i] += 1.0
                    reward[i] += components['precision_reward'][i]
            
            # Add fast-paced maneuvers reward
            player_speed = np.linalg.norm(o['right_team_direction'][o['active']])
            if player_speed > 0.5:
                components['pace_reward'][i] += 0.5
                reward[i] += components['pace_reward'][i]
                
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
                self.sticky_actions_counter[i] += int(action)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
