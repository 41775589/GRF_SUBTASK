import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that emphasizes fast-paced attacking football through specialized dense reward signals aimed at
    enhancing offensive maneuvers and precision-based finishing skills.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._offensive_zones = 5
        self._goal_approach_reward = 0.2
        self._goal_zone_threshold = [(i+1)*0.2 for i in range(self._offensive_zones)]
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            if obs['game_mode'] != 0:
                continue  # No rewards when game is in special modes

            dist_to_goal = 1 - abs(obs['ball'][0])

            for i, threshold in enumerate(self._goal_zone_threshold):
                if dist_to_goal > threshold:
                    # Reward attacking maneuvers that advance into opponent's field closer to the goal
                    components["offensive_reward"][rew_index] += self._goal_approach_reward * (1 + i * 0.5)
                    break

            reward[rew_index] += components["offensive_reward"][rew_index]
        
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
