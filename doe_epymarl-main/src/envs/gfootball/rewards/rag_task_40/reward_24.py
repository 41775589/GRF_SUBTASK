import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive plays and counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defense_position_reward = 0.5
        self.counterattack_bonus = 0.3

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
        # Initialize reward components
        components = {"base_score_reward": reward.copy(),
                      "defense_position_reward": [0.0, 0.0],
                      "counterattack_reward": [0.0, 0.0]}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        # Extended reward logic for defensive actions and counterattacks
        for rew_index, o in enumerate(observation):
            # Adapt defensive behavior by positioning
            if o['ball_owned_team'] == 1:  # If opponent team has the ball
                player_pos = o['left_team'][o['active']]
                goal_pos = [-1, 0]  # Position of left team's goal
                player_goal_dist = np.linalg.norm(goal_pos - player_pos)
                if player_goal_dist < 0.3:  # Close to goal acts as a better defensive position
                    components["defense_position_reward"][rew_index] = self.defense_position_reward
                    reward[rew_index] += components["defense_position_reward"][rew_index]

            # Encourage counterattacks
            if o['ball_owned_team'] == 0 and o['game_mode'] == 0:  # Normal mode and ball owned by agent's team
                player_pos = o['left_team'][o['active']]
                opponent_goal_pos = [1, 0]  # Position of opponent's goal
                player_goal_dist = np.linalg.norm(opponent_goal_pos - player_pos)
                if player_goal_dist < 0.5:  # Moving the ball towards opponent's goal
                    components["counterattack_reward"][rew_index] = self.counterattack_bonus
                    reward[rew_index] += components["counterattack_reward"][rew_index]

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
