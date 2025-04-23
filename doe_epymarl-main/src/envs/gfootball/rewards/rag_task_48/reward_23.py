import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes agents for executing successful high passes from midfield to create direct scoring opportunities.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.high_pass_effectiveness = 0.2  # Reward weight for effective high passes
        self.scoring_opportunity_created = 1.0  # Reward weight for creating scoring opportunities
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', {}).get('sticky_actions_counter', np.zeros(10))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the controlled player is midfield and execute a high pass
            if o['active'] and o['left_team_roles'][o['active']] in [4, 5, 6, 7]:  # Midfield roles
                if o['sticky_actions'][3] == 1 and o['ball'][2] > 0.15:  # Assuming index '3' is for high pass action and z > 0.15 implies high pass
                    # Calculate if this leads to a teammate near scoring position
                    teammates_positions = o['left_team']
                    ball_travel_destination = o['ball'][:2] + o['ball_direction'][:2] * 5  # Predicted future position of the ball
                    
                    for teammate_pos in teammates_positions:
                        if np.linalg.norm(teammate_pos - ball_travel_destination) < 0.1 and np.abs(teammate_pos[0]) > 0.7:  # Close to goal area and close to ball
                            components['high_pass_reward'][rew_index] = self.high_pass_effectiveness + self.scoring_opportunity_created
                    
            reward[rew_index] += components['high_pass_reward'][rew_index]
        
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
