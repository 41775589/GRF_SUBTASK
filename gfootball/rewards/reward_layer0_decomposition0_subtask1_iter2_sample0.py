import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward tailored for enhancing defensive play, teamwork, and passing strategy."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward parameters
        self.defensive_zone_reward = 0.2
        self.ball_positioning_reward = 0.1
        self.passing_reward = 0.05
        self.loose_ball_capture_reward = 0.15

    def reset(self):
        """Reset the environment and reset the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add current object states to the pickle to save the state."""
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve state from the pickle and set the current state."""
        from_pickle = self.env.set_state(state)
        picked_states = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = picked_states.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Customize the reward based on defensive play, ball positioning, and passive interception."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_reward": [0.0] * len(reward),
            "ball_positioning_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
            "loose_ball_capture_reward": [0.0] * len(reward)
        }

        for idx, obs in enumerate(observation):
            ball_position = obs.get('ball')
            active_player_position = obs['left_team'][obs['active']] if obs.get('ball_owned_team') == 0 else obs['right_team'][obs['active']]
            ball_owner = obs.get('ball_owned_player')
            
            # Defending zone reward: being close to own goal when opponent has the ball
            player_team_goal_x = -1 if obs.get('ball_owned_team') == 1 else 1
            if np.linalg.norm(active_player_position - [player_team_goal_x, 0]) < 0.3:
                components["defensive_zone_reward"][idx] = self.defensive_zone_reward
                reward[idx] += components["defensive_zone_reward"][idx]
            
            # Ball positioning reward: encouraging safe passing and catching in crucial areas
            if ball_owner == obs['active']:
                if abs(ball_position[0]) > 0.8:  # near the goals
                    components["ball_positioning_reward"][idx] = self.ball_positioning_reward
                    reward[idx] += components["ball_positioning_reward"][idx]
            
            # Passing reward: reward completed passes
            if 'action' in obs:
                if obs['action'] == 'pass' and ball_owner == obs['active']:
                    components["passing_reward"][idx] = self.passing_reward
                    reward[idx] += components["passing_reward"][idx]
            
            # Reward capturing of a loose ball that turns possession
            if obs.get('ball_owned_team') == -1:
                distance_to_ball = np.linalg.norm(active_player_position - ball_position[:2])
                if distance_to_ball < 0.1:
                    components["loose_ball_capture_reward"][idx] = self.loose_ball_capture_reward
                    reward[idx] += components["loose_ball_capture_reward"][idx]

        return reward, components

    def step(self, action):
        """Execute environment step and augment reward and info with custom logic."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
