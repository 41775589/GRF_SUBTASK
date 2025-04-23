import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper encourages offensive strategies by rewarding accurate shooting, effective dribbling,
    and using various passing techniques.
    """
    def __init__(self, env):
        super().__init__(env)
        self.shoot_distance_threshold = 0.2  # Threshold for considering shooting towards goal
        self.pass_types_threshold = 0.4      # Distance threshold for considering a long/high pass
        self.dribble_reward_scale = 0.1      # Scaling factor for dribbling rewards

        # Keeping counters of actions for the duration of the game
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the environment and counters """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Returns the current state of the environment including rewards custom states """
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist(),
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Sets the state of the environment from a saved state """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """ Custom reward function to enhance offensive strategy training """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward
        
        # Initialize components
        components = {
            "base_score_reward": reward.copy(),
            "shoot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            # Component: Shooting towards goal
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                dist_to_goal = abs(o['ball'][0] - 1)  # Distance from the right goal
                if dist_to_goal < self.shoot_distance_threshold:
                    components["shoot_reward"][rew_index] = 1.0 - dist_to_goal

            # Component: Dribbling Effectiveness
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is dribbling
                components["dribble_reward"][rew_index] += self.dribble_reward_scale

            # Component: Effective Passing
            if o['sticky_actions'][0:8].sum() > 0:  # Assuming indexes 0-7 cover directional movements
                ball_move_dist = np.linalg.norm(o['ball_direction'][0:2])
                if ball_move_dist > self.pass_types_threshold:
                    components["pass_reward"][rew_index] += ball_move_dist

            # Aggregate all rewards for this step
            total_reward = sum([components[comp][rew_index] for comp in components])
            reward[rew_index] += total_reward

        return reward, components

    def step(self, action):
        """ Steps the environment, calculates and applies rewards """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add component values to info for logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky action counters
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
