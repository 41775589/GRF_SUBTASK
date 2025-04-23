import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for shooting accuracy from central field positions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.central_zone_threshold = 0.2  # Distance from the center (y-axis) which defines the central field.
        self.goal_x_coordinate = 1.0  # x-coordinate for right goal (assuming shooting left to right).
        self.active_player_key = 'active'
        self.ball_pos_key = 'ball'
        self.ball_own_team_key = 'ball_owned_team'
        self.goal_area_threshold = 0.05  # Distance near goal to be considered close
        self.shooting_reward = 1.0  # Reward for shooting from central field
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
        components = {"base_score_reward": reward.copy()}
        
        # Initialize the new rewards structure
        shooting_rewards = [0.0] * len(reward)
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            
            # Determine if the active player is in the central zone and has the ball
            if ((abs(obs[self.ball_pos_key][1]) <= self.central_zone_threshold) and
                (obs[self.ball_own_team_key] == 0) and 
                (obs[self.active_player_key] == obs['ball_owned_player'])):
                
                # Check if the player is close to the opponent's goal
                if abs(self.goal_x_coordinate - obs[self.ball_pos_key][0]) < self.goal_area_threshold:
                    shooting_rewards[rew_index] = self.shooting_reward
                    
        # Update total rewards with shooting reward
        reward = [rew_i + shoot_rew_i for rew_i, shoot_rew_i in zip(reward, shooting_rewards)]
        components["shooting_rewards"] = shooting_rewards

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
