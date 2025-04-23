import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_zones = {
            "high_pressure": 0.2,  # Close to opponent players
            "medium_pressure": 0.4
        }
        self.pass_rewards = {
            "short_pass": 0.05,
            "long_pass": 0.1,
            "high_pass": 0.1
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Initialize the components dictionary
        components = {"base_score_reward": reward.copy(),
                      "pressure_pass_reward": [0.0, 0.0]}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        # Iterate through observations for each agent
        for i in range(len(reward)):
            # Extract observation of agent i
            o = observation[i]
            
            if o['game_mode'] in [0, 1] and o['ball_owned_team'] == 0:
                active_player_pos = o['left_team'][o['active']]
                opponents = o['right_team']
                min_distance = np.min(np.linalg.norm(opponents - active_player_pos, axis=1))
                
                # Check if under high pressure and made a pass
                if min_distance < self.pressure_zones["high_pressure"]:
                    pass_type = self._determine_pass_type(o)
                    if pass_type:
                        components["pressure_pass_reward"][i] += self.pass_rewards[pass_type]
                        reward[i] += components["pressure_pass_reward"][i]
                
                # Moderate pressure scenario
                elif min_distance < self.pressure_zones["medium_pressure"]:
                    pass_type = self._determine_pass_type(o)
                    if pass_type:
                        components["pressure_pass_reward"][i] += self.pass_rewards[pass_type] * 0.5
                        reward[i] += components["pressure_pass_reward"][i]

        return reward, components

    def _determine_pass_type(self, observation):
        if observation['sticky_actions'][6]:  # Long pass
            return "long_pass"
        elif observation['sticky_actions'][7]:  # High pass
            return "high_pass"
        elif observation['sticky_actions'][5]:  # Short pass
            return "short_pass"
        return None

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
