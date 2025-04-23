import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a precision reward for making accurate shots in close range.
    This is specifically useful for teaching agents to handle tight space maneuvers and adjusting shot 
    power and angle effectively to score past the goalkeeper.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_zone_threshold = 0.2  # Close range threshold relative to x-axis of the goal
        self.precision_reward = 0.2  # Reward for shooting within the goal threshold

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
        components = {'base_score_reward': reward.copy(), 'precision_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the player is close to the opponent's goal and has taken a shot
            player_pos = o['right_team'][o['active']]
            ball_pos = o['ball'][:2]  # we only need x, y coordinates
            is_shot_taken = 'action_shot' in o['sticky_actions'] and o['sticky_actions']['action_shot']

            if player_pos[0] > (1 - self.goal_zone_threshold) and is_shot_taken:
                # Calculate the distance of the ball from the goal line after taking a shot
                if np.abs(ball_pos[1]) < 0.044:  # y-axis value of the goal boundaries
                    # Reward based on how close the ball is to the center of the goal
                    goal_center_y = 0
                    distance_to_goal_center = np.abs(ball_pos[1] - goal_center_y)
                    components['precision_reward'][rew_index] = self.precision_reward * (1 - distance_to_goal_center / 0.044)
                    reward[rew_index] += components['precision_reward'][rew_index]

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
