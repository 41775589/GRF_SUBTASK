import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on effective clearance from the defensive zone under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define key regions for defensive clearance
        self.defensive_threshold = -0.4  # x-coordinate threshold for defining 'defensive' zone
        self.pressure_distance = 0.2   # distance to check for nearby opponents to consider 'under pressure'
        self.clearance_success_reward = 1.0  # reward for successful clearance

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
                      "clearance_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            my_team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'
            opponent_team = 'right_team' if my_team == 'left_team' else 'left_team'
            
            # Get positions and ownership information
            ball_position = o['ball'][:2]  # only consider x, y
            player_position = o[my_team][o['active']][:2]
            opponents_positions = o[opponent_team]
            ball_owned = o['ball_owned_team'] == 0 if my_team == 'left_team' else o['ball_owned_team'] == 1
            
            # Calculate the pressure situation and if the ball is in the defensive zone
            in_defensive_zone = ball_position[0] <= self.defensive_threshold if my_team == 'left_team' else ball_position[0] >= -self.defensive_threshold
            under_pressure = any(np.linalg.norm(player_position - opponent[:2]) < self.pressure_distance for opponent in opponents_positions)

            # Check for clearance success condition: ball moves from the defensive zone to a safer area
            if in_defensive_zone and under_pressure and ball_owned:
                previous_ball_position = player_position - o['ball_direction'][:2]  # estimate the previous ball position
                moved_out_of_defense = previous_ball_position[0] > self.defensive_threshold if my_team == 'left_team' else previous_ball_position[0] < -self.defensive_threshold
                if moved_out_of_defense:
                    components["clearance_reward"][rew_index] = self.clearance_success_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]
        
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
