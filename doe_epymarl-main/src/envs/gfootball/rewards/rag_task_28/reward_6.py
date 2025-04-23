import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a structured dribbling and feinting skill reward around the goalkeeper area."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_threshold = 0.1
        self.close_to_goal_bonus = 0.2
        self.dribbling_bonus = 0.1
        self.under_pressure_bonus = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "dribbling_bonus": [0.0] * len(reward),
                      "close_to_goal_bonus": [0.0] * len(reward),
                      "under_pressure_bonus": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 0:  # Check if the ball is owned by the controlled team
                distance_to_goal = abs(o['ball'][0] - 1)
                if distance_to_goal < self.distance_threshold:
                    # Reward for being close to goal while possessing the ball
                    components['close_to_goal_bonus'][i] = self.close_to_goal_bonus
                    reward[i] += components['close_to_goal_bonus'][i]
                
                # Additional dribbling bonus when dribbling action is in sticky actions
                if o['sticky_actions'][9] == 1:  # Dribble action index
                    components['dribbling_bonus'][i] = self.dribbling_bonus
                    reward[i] += components['dribbling_bonus'][i]
                
                # Pressure situation - multiple opponents close by
                opponent_distances = np.sqrt(np.sum(np.square(o['right_team'] - o['ball'][:2]), axis=1))
                close_opponents = np.sum(opponent_distances < self.distance_threshold)
                if close_opponents >= 2:  # If at least two opponents are very close
                    components['under_pressure_bonus'][i] = self.under_pressure_bonus
                    reward[i] += components['under_pressure_bonus'][i]

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
