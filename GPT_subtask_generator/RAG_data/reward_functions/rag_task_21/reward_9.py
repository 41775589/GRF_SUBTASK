import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focused on enhancing defensive skills, specifically 
    responsiveness and interception abilities in high-pressure situations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to define interception and pressure zones
        self.defensive_zones = np.linspace(-1.0, 1.0, num=5)  # Dividing the field into defensive zones
        self.interception_bonus = 0.1  # Reward for successful interception in a high-pressure zone
        
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
        components = {"base_score_reward": reward.copy(), "interception_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["interception_bonus"][rew_index] = self.evaluate_defensive_play(o)

            reward[rew_index] += components["interception_bonus"][rew_index]
        
        return reward, components

    def evaluate_defensive_play(self, obs):
        """
        Evaluates and rewards defensive plays based on player's position,
        ball possession, and pressure zones.
        """
        interception_reward = 0.0
        if obs['ball_owned_team'] == 1 and obs['active'] in obs['left_team']:
            player_pos = obs['left_team'][obs['active']]
            ball_pos = obs['ball'][:2]
            
            # Check if player is within a defensive zone and close to the ball
            if any(zone - 0.1 < player_pos[0] < zone + 0.1 for zone in self.defensive_zones):
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                if distance_to_ball < 0.2:
                    interception_reward = self.interception_bonus
        return interception_reward
    
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
