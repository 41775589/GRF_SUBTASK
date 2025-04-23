import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for effective ball clearance from the defensive zone under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_bonus = 0.5  # Reward given for effective clearance
        self.defensive_zone_threshold = -0.5  # X-coordinates threshold for defensive zone (normalized)
        self.pressure_distance_threshold = 0.2  # Distance threshold to consider opponents putting pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardsWrapper'] = {}  # Add necessary state components if any
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            
            # Ball in defensive zone and owned by agent's team
            if ball_position[0] <= self.defensive_zone_threshold and o['ball_owned_team'] == 0:
                opponents = o['right_team']
                
                # Check if opponents are close enough to apply pressure
                under_pressure = any(np.linalg.norm(opponent - player_position) < self.pressure_distance_threshold for opponent in opponents)
                
                if under_pressure:
                    action_effectiveness = np.linalg.norm(ball_position - player_position)
                    
                    # Bonus for moving the ball significantly away from defensive zone under pressure
                    if action_effectiveness > self.pressure_distance_threshold:
                        components["clearance_bonus"][rew_index] = self.clearance_bonus
                        reward[rew_index] += components["clearance_bonus"][rew_index]

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
