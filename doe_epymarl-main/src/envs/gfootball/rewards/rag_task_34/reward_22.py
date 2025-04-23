import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes mastering close-range attacks, 
    precision shooting, and effective dribbling against goalkeepers.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for controlling the specificity of the rewards
        self.goal_area_distance_threshold = 0.2
        self.possession_bonus = 0.1
        self.shoot_close_range_bonus = 0.5

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
        components = {
            "base_score_reward": reward.copy(),
            "possession_bonus": [0.0] * len(reward),
            "close_range_shoot_bonus": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
  
        for rew_index, o in enumerate(observation):
            player_pos = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            ball_pos = o['ball']
            distance_to_goal = np.abs(ball_pos[0] - 1)  # Assuming attacking towards right goal
            
            # Close to goal area and ball possession reward
            if distance_to_goal < self.goal_area_distance_threshold and o['ball_owned_team'] == 1:
                components["possession_bonus"][rew_index] = self.possession_bonus
                if 'action_bottom_right' in o['sticky_actions']:  # Assuming shooting action
                    components["close_range_shoot_bonus"][rew_index] = self.shoot_close_range_bonus
            reward[rew_index] += components["possession_bonus"][rew_index] + components["close_range_shoot_bonus"][rew_index]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
