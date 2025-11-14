import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive action-based reward for two soccer agents focused on defensive capabilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)
        
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_reward": [0] * len(reward)}
        
        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            base_reward = reward[rew_index]
            
            # Reward for successful defensive actions
            successful_defense_reward = 0
            if 'ball_owned_team' in player_obs and player_obs['ball_owned_team'] == 0:
                # Assuming ball possession is a good direct defensive outcome
                successful_defense_reward += 0.5
            
            # Additional positive reinforcement for moving towards an opponent with the ball
            if 'ball' in player_obs and 'right_team' in player_obs:
                ball_pos = player_obs['ball'][:2]
                player_pos = player_obs['right_team'][player_obs['active']][:2]
                distance = np.linalg.norm(ball_pos - player_pos)
                
                # Closer the player to the ball, higher the potential interception reward
                distance_reward = min(0.3, 1.0 / (1.0 + distance))
                successful_defense_reward += distance_reward
            
            components['defensive_action_reward'][rew_index] = successful_defense_reward
            reward[rew_index] = base_reward + successful_defense_reward
        
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
