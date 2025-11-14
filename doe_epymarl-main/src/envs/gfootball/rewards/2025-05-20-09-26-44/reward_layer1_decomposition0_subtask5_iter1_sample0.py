import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances passing and dribbling in line with agent's offensive strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for various actions
        self.ball_control_reward = 0.05
        self.pass_success_reward = 0.2
        self.dribbling_skill_reward = 0.1
        self.player_pos_X_threshold = 0.5

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', {'sticky_actions_counter': np.zeros(10)}).get('sticky_actions_counter', np.zeros(10))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control": [0.0],
            "successful_passing": [0.0],
            "dribbling_skill": [0.0]
        }

        if observation is None:
            return reward, components
        
        agent_obs = observation[0]  # Single agent scenario

        # Encourage ball control in critical areas
        if agent_obs['ball_owned_team'] == 1 and agent_obs['ball'][0] > self.player_pos_X_threshold: # Right team and forward X field
            components["ball_control"][0] += self.ball_control_reward
            reward[0] += components["ball_control"][0]

        # Reward successful passes that change the game mode to KickOff, meaning a successful reception
        if 'game_mode' in agent_obs and agent_obs['game_mode'] == 1:  # KickOff mode
            components["successful_passing"][0] += self.pass_success_reward
            reward[0] += components["successful_passing"][0]

        # Check for effective dribbling
        if 'sticky_actions' in agent_obs and agent_obs['sticky_actions'][9]:  # Dribble action
            components["dribbling_skill"][0] += self.dribbling_skill_reward
            reward[0] += components["dribbling_skill"][0]

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
