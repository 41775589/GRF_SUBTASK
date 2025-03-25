import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper focuses on enhancing defensive coordination. It adds
    rewards for maintaining proper position relative to the ball and each other
    near their penalty area under high-pressure scenarios.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalty_area_threshold_x = 0.7  # x-coordinate threshold for near penalty area
        self.ball_proximity_reward = 0.05
        self.agent_proximity_reward = 0.1
        self.defensive_formation_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_data'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Assuming checkpoint data is not transient and should be restored
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "ball_proximity_reward": [0.0] * len(reward),
            "agent_proximity_reward": [0.0] * len(reward),
            "defensive_formation_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for index, rew in enumerate(reward):
            obs_agent = observation[index]
            
            # Reward for being close to the ball in the defensive area
            ball_position = obs_agent['ball']
            if obs_agent['ball_owned_team'] == 1 and ball_position[0] > self.penalty_area_threshold_x:
                distance_to_ball = np.linalg.norm(ball_position - obs_agent['left_team'][obs_agent['active']])
                if distance_to_ball < 0.1:  # Close to the ball in a defensive scenario
                    components["ball_proximity_reward"][index] = self.ball_proximity_reward
                    rew += components["ball_proximity_reward"][index]

            # Additional reward if the agents maintain good relative positions defensively
            if len(observation) > 1 and np.abs(observation[0]['left_team'][0][0] - observation[1]['left_team'][0][0]) < 0.05:
                for r in range(len(reward)):
                    components["agent_proximity_reward"][r] = self.agent_proximity_reward
                    reward[r] += components["agent_proximity_reward"][r]
                    
            # Forming a defensive line or formation
            if len(observation) > 1 and np.abs(observation[0]['left_team'][0][1] - observation[1]['left_team'][0][1]) < 0.1:
                components["defensive_formation_reward"][index] = self.defensive_formation_reward
                rew += components["defensive_formation_reward"][index]
            
            reward[index] = rew

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
