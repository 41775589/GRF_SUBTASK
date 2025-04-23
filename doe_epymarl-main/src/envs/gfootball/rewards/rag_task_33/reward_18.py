import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a specialized reward for executing long-distance shots.
    It encourages agents to beat defenders and shoot from outside the penalty box.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalty_box_x_threshold = 0.7  # Threshld for considering a position outside the penalty box
        
        # Each segment outside the penalty box adds to the reward, promoting long shots.
        self.outside_penalty_box_reward = 0.5

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
                      "long_shot_reward": [0.0] * len(reward)}
        
        # Stop processing if observation is none
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation), "Mismatch in reward and observation lengths"
        
        for agent_index, agent_reward in enumerate(reward):
            agent_obs = observation[agent_index]
            
            # Long shot condition: ball owned by current team, outside the penalty box, and moving towards the goal
            if (agent_obs['ball_owned_team'] == agent_obs['right_team'] and \
                agent_obs['ball'][0] > self.penalty_box_x_threshold and \
                agent_obs['ball_direction'][0] > 0):  # Assuming right direction as towards opponent's goal

                components['long_shot_reward'][agent_index] = self.outside_penalty_box_reward
                reward[agent_index] += components['long_shot_reward'][agent_index] * (1 + np.clip(agent_obs['ball'][0], 0, 1))
        
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
