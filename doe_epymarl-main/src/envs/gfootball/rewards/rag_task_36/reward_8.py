import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for dribbling and dynamic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.threshold_for_positioning = 0.3
        self.dribble_reward = 0.1
        self.non_dribble_penalty = -0.05
        self.positioning_reward = 0.2
        self.position_achieved = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.position_achieved = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "position_achieved": self.position_achieved
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_achieved = from_pickle['CheckpointRewardWrapper']['position_achieved']
        return from_pickle

    def reward(self, reward):
        """Calculate augmented reward."""
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        dribble_reward = 0.0
        positioning_reward = 0.0
        
        for rew_index, o in enumerate(observation):
            # Handling dribbling actions
            if o['sticky_actions'][9]:  # 'dribble' action is indexed at 9
                dribble_reward += self.dribble_reward
            else:
                dribble_reward += self.non_dribble_penalty
            
            # Rewarding dynamic positioning
            if not self.position_achieved:
                distance_to_goal = np.linalg.norm([o['right_team'][0][0] - 1, o['right_team'][0][1]])
                if distance_to_goal < self.threshold_for_positioning:
                    positioning_reward += self.positioning_reward
                    self.position_achieved = True  # Prevent further rewards in this episode

        updated_reward = [base_reward[i] + dribble_reward + (positioning_reward if self.position_achieved else 0)
                          for i in range(len(reward))]
        reward_components = {
            "base_score_reward": base_reward,
            "dribble_reward": dribble_reward,
            "positioning_reward": positioning_reward
        }
        return updated_reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value if isinstance(value, list) else [value])
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state

        return observation, reward, done, info
