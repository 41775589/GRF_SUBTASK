import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive and strategic actions."""

    def __init__(self, env):
        super().__init__(env)
        # Counter for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize defensive position checkpoints
        self._num_defensive_checkpoints = 5
        self._defensive_checkpoint_reward = 0.05
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any specific state if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_team' not in o:
                continue
            
            is_own_team = (o['ball_owned_team'] == 0)
            opponent_approaching = self._detect_opponent_approaching(o)

            # Add rewards for maintaining strategic positions during defensive scenarios
            if is_own_team and opponent_approaching:
                def_pos_reward = self._evaluate_defensive_position(o)
                components["defensive_position_reward"][rew_index] += def_pos_reward
                reward[rew_index] += components["defensive_position_reward"][rew_index]

        return reward, components

    def _detect_opponent_approaching(self, obs):
        # Simulate a function that detects if an opponent is moving towards player's goal
        team_index = 1 if obs['ball_owned_team'] == 0 else 0
        velocities = obs[f'{["right", "left"][team_index]}_team_direction']
        
        goal_dir = np.sign(obs['ball'][0])
        if goal_dir == 0:
            return False
        
        # Detect if any players are moving towards the goal
        approaching = np.any(velocities[:, 0] * goal_dir < 0)
        return approaching

    def _evaluate_defensive_position(self, obs):
        # Placeholder function logic to evaluate and score defensive positioning
        my_team = obs['left_team'] if obs['ball_owned_team'] == 1 else obs['right_team']
        goal_x = -1 if obs['ball_owned_team'] == 1 else 1
        distances = np.linalg.norm(my_team - np.array([goal_x, 0]), axis=-1)
        # Reward closer positions more
        closer_positions_reward = np.mean(1 / (1 + distances))
        return self._defensive_checkpoint_reward * closer_positions_reward

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
