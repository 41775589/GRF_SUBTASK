import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards the use of Stop-Dribble under pressure tactically."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_importance = 0.1   # Reward for maintaining control under pressure
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_data = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = np.array(saved_data.get('sticky_actions_counter', []), dtype=int)
        return from_pickle

    def reward(self, reward):
        # Initial components breakdown
        components = {"base_score_reward": reward.copy(),
                      "control_under_pressure_reward": [0.0] * len(reward)}
        
        # Get current observations from the environment
        observation = self.env.unwrapped.observation()
        
        for agent_idx in range(len(reward)):
            # Process observation for individual agents
            agent_obs = observation[agent_idx]
            is_under_pressure = self.is_agent_under_pressure(agent_obs)
            if is_under_pressure:
                # Reward is increased if the agent is under pressure and performs a stop dribble
                stop_dribble_action = agent_obs['sticky_actions'][9]  # Assuming index 9 is a stop dribble action.
                if stop_dribble_action == 1:
                    components["control_under_pressure_reward"][agent_idx] = self.ball_control_importance
                    reward[agent_idx] += components["control_under_pressure_reward"][agent_idx]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def is_agent_under_pressure(self, agent_obs):
        # Example condition: Consider agent under pressure if close to opponent players
        agent_position = agent_obs['right_team'] if agent_obs['left_team_active'][agent_obs['active']] else agent_obs['left_team']
        opponents = agent_obs['left_team'] if agent_obs['left_team_active'][agent_obs['active']] else agent_obs['right_team']
        distance_to_opponents = np.array([np.linalg.norm(opponent - agent_position[agent_obs['active'], :]) for opponent in opponents])
        close_opponents = np.sum(distance_to_opponents < 0.1)  # Arbitrary threshold to determine "pressure"
        return close_opponents > 0  # Under pressure if one or more opponents are too close
