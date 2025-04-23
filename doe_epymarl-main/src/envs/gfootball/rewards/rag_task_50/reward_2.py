import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward for executing accurate long passes, promoting vision, timing and precision."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_previous_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_previous_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'ball_previous_position': self.ball_previous_position}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_previous_position = from_pickle['CheckpointRewardWrapper']['ball_previous_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_reward': reward}

        assert len(reward) == 2  # Assuming 2 teams
        
        components = {"base_score_reward": reward.copy(), 
                      "long_pass_reward": [0.0, 0.0]}
        
        # Check for long passes
        ball_current_position = observation[0]['ball'][:2]  # Take only x, y position
        if self.ball_previous_position is not None:
            # Euclidean distance between two positions to assess 'long pass'
            distance = np.linalg.norm(np.array(ball_current_position) - np.array(self.ball_previous_position))
            minimum_long_pass_distance = 0.3  # Threshold for a pass to be considered 'long'
            
            # Determine if it was a pass based on ball ownership changing and distance traveled
            if (observation[0]['ball_owned_team'] != observation[1]['ball_owned_team']) and \
               distance >= minimum_long_pass_distance:
                for team in range(2):
                    if observation[team]['ball_owned_team'] == -1:  # Ball transitioned, check previous owner
                        if self.ball_previous_position_owner == team:
                            components['long_pass_reward'][team] = distance * 0.1  # Reward proportional to distance
                            reward[team] += components['long_pass_reward'][team]

        # Updating ball position and its owner
        self.ball_previous_position = ball_current_position
        self.ball_previous_position_owner = observation[0]['ball_owned_team']
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add component details to info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        # Create action counts in info
        for i in range(len(self.sticky_actions_counter)):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
