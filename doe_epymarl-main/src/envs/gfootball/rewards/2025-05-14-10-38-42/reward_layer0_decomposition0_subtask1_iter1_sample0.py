import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward tailored for enhancing defensive play, ball control,
    and passing in a team-oriented and disruptive strategy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define rewards for different aspects of the game
        self.defensive_positioning_reward = 0.1
        self.ball_control_reward = 0.05
        self.pass_reward = 0.05

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
        """Customize the reward based on defensive play, ball control, and team passes."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_positioning_reward": [0.0] * len(reward),
            "ball_control_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }

        for idx, obs in enumerate(observation):
            ball_owned_team = obs.get('ball_owned_team')  # -1: none, 0: left, 1: right
            ball_owned_player = obs.get('ball_owned_player')
            active_player = obs.get('active')

            # Ball control incentive
            if ball_owned_team == 0 and ball_owned_player == active_player:
                components["ball_control_reward"][idx] = self.ball_control_reward
                reward[idx] += components["ball_control_reward"][idx]

            # Defensive positioning incentive
            # Check if the player is closer to defensive goal when the ball is near
            player_position = obs['left_team'][active_player] if obs['active'] in obs['left_team'] else obs['right_team'][active_player]
            goal_position = [-1, 0] if obs['active'] in obs['left_team'] else [1, 0]
            if np.linalg.norm(player_position - goal_position) < np.linalg.norm(obs['ball'][:2] - goal_position):
                components["defensive_positioning_reward"][idx] = self.defensive_positioning_reward
                reward[idx] += components["defensive_positioning_reward"][idx]

            # Passing reward
            # Check the last action was a successful pass leading to control gain
            if ball_owned_team == 0 and 'action_id' in obs:  # Assuming action_id for pass is correctly configured
                if obs['action_id'] == 'pass' and ball_owned_player == active_player:
                    components["pass_reward"][idx] = self.pass_reward
                    reward[idx] += self.pass_reward

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
