import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning reward based on agents' movements and game modes."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_checkpoints_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.positional_checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positional_checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['game_mode'] != 0:  # Check if not in normal play mode
                # Apply strategic rewards in special game modes
                components['positional_reward'][rew_index] = 0.05
                reward[rew_index] += components['positional_reward'][rew_index]

            # Assess positioning reward based on team's current strategic need
            team_has_ball = (o['ball_owned_team'] == 0) if 'ball_owned_team' in o else False
            if team_has_ball:
                ball_position = o['ball'][:2] if 'ball' in o else [0, 0]
                if ball_position[0] > 0.5:  # Ball in opponent's half
                    pass_section = 3 if abs(ball_position[1]) < 0.2 else 2
                else:
                    pass_section = 1
                
                checkpoint_key = (rew_index, pass_section)
                if checkpoint_key not in self.positional_checkpoints_collected:
                    self.positional_checkpoints_collected[checkpoint_key] = True
                    components['positional_reward'][rew_index] = 0.1
                    reward[rew_index] += components['positional_reward'][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
