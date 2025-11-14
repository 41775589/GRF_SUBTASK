import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward based on advanced dribbling and shooting decisions near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_reward_multiplier = 1.3
        self.shooting_reward_multiplier = 2.0
        self.goal_zone_threshold = 0.3
        self.height_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "dribbling_reward": [0.0], 
            "shooting_reward": [0.0],
            "height_advantage": [0.0]
        }

        if observation is None:
            return reward, components

        o = observation[0]
        if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
            ball_position = np.array(o['ball'][:2])
            goal_position = np.array([1.0, 0])  # Assumption of right goal
            distance_to_goal = np.linalg.norm(ball_position - goal_position)

            # Dribbling reward for maintaining ball possession while moving forward
            if o['sticky_actions'][9] == 1:  # Assuming index 9 corresponds to 'dribble'
                components["dribbling_reward"][0] = self.dribbling_reward_multiplier * (1 - distance_to_goal)
                reward[0] += components["dribbling_reward"][0]

            # Shooting reward near the goal and considering vertical angle
            if distance_to_goal < self.goal_zone_threshold:
                height_advantage = self.height_factor * abs(ball_position[1])
                components["height_advantage"][0] = height_advantage
                if o['sticky_actions'][5] == 1 or o['sticky_actions'][6] == 1:  # Assuming indices for shooting actions
                    components["shooting_reward"][0] = self.shooting_reward_multiplier + height_advantage
                    reward[0] += components["shooting_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = observation if isinstance(observation, list) else [observation]
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action_state in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action_state
                    info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
