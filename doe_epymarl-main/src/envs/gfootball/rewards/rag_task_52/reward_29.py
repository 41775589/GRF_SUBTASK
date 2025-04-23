import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward system to enhance defending strategies by focusing on specialized training 
    modules for tackling proficiency, movement control and pressured passing tactics.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 0.05
        self._movement_control_reward = 0.05
        self._pressured_pass_reward = 0.1

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_control_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o['ball_owned_team']
            
            # Reward for successful tackles
            if 'ball_owned_team' in o and ball_owned_team == 0:  # assuming 0 is the team index for the agent's team
                if o['game_mode'] in {3, 6}:  # representing FreeKick or Penalty modes which might imply loss of the ball due to a tackle
                    components['tackle_reward'][rew_index] = self._tackle_reward
                    reward[rew_index] += components['tackle_reward'][rew_index]
            
            # Reward for efficient movement control
            if o['ball_owned_team'] == 0:  # Ball is with the team
                player_speed = np.linalg.norm(o['left_team_direction'][o['active']])
                # Lower speed might imply more control; hence rewarding stopping or slower speeds
                if player_speed < 0.01:  # Threshold for speed that indicates stopping or very slow movement
                    components['movement_control_reward'][rew_index] = self._movement_control_reward
                    reward[rew_index] += components['movement_control_reward'][rew_index]
            
            # Reward for pressured passes
            # Given a pass situation under opponent's presence
            if ball_owned_team == 0 and o['game_mode'] == 3:  # FreeKick mode assumed to be result of successful pressured pass
                components['pressured_pass_reward'][rew_index] = self._pressured_pass_reward
                reward[rew_index] += components['pressured_pass_reward'][rew_index]
                
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
