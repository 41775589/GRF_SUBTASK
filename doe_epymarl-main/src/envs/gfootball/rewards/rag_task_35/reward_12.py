import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning reward based on player movement and positioning."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoint_counter = {}

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoint_counter = {}
        return self.env.reset(**kwargs)
    
    def get_state(self, to_pickle):
        to_pickle['position_checkpoint_counter'] = self.position_checkpoint_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_checkpoint_counter = from_pickle['position_checkpoint_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components
        
        for idx, obs in enumerate(observation):
            # distance to ball
            ball_position = np.array(obs['ball'][:2])
            player_position = np.array(obs['left_team'][obs['active']] if obs['left_team_active'][obs['active']] else obs['right_team'][obs['active']])
            
            # calculate distance to ball
            distance_to_ball = np.linalg.norm(ball_position - player_position)
            
            # Movement directives encourages strategic positioning
            movement_reward = 0.1 * (1 - min(distance_to_ball, 1))
            
            # Encourage changing active player to adapt strategy
            player_role = obs['left_team_roles'][obs['active']] if obs['left_team_active'][obs['active']] else obs['right_team_roles'][obs['active']]
            if player_role in [4, 5, 8]:  # Midfield or attacking midfield roles
                movement_reward *= 1.5
            
            self.sticky_actions_counter += obs['sticky_actions']  # Count sticky actions used by each player
            action_frequency_reward = -0.01 * np.sum(self.sticky_actions_counter)  # discourage frequent sticky actions

            # Incorporate to total reward
            reward[idx] += movement_reward + action_frequency_reward
            components.setdefault("movement_reward", []).append(movement_reward)
            components.setdefault("action_frequency_reward", []).append(action_frequency_reward)
        
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
