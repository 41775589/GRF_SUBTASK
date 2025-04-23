import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that is designed to encourage mastering high passes and effective 
    positioning by wide midfield players. The aim is to expand the play field,
    supporting lateral transitions and stretching the opposition's defense. 
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Holds counts of sticky actions used.
        self.high_pass_reward = 0.05
        self.positioning_reward = 0.03
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": 0.0,
                      "positioning_reward": 0.0}

        if observation is None:
            return reward, components
        
        # Check for high passes and wide midfielder positioning
        if "sticky_actions" in observation and "left_team_roles" in observation:
            if observation["sticky_actions"][9] == 1:  # Index 9 for high pass action
                components["high_pass_reward"] += self.high_pass_reward
                reward += self.high_pass_reward

            # Encourage proper positioning of the specific role "LM" or "RM" in the side sectors
            if observation['right_team_roles'][self.env.unwrapped.player] in [6, 7]:  # 6 = LM, 7 = RM
                player_pos = observation['right_team'][self.env.unwrapped.player]
                # Reward 'LM' or 'RM' being in the wide areas of the field
                if abs(player_pos[1]) > 0.3:  # Y position greater than 0.3 either side
                    components['positioning_reward'] += self.positioning_reward
                    reward += self.positioning_reward

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Update with final reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
