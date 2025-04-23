import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic rewards focused on passing, shooting, and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Tactical objectives for goal proximity, accurate passes, and strategic decision-making
        self.goal_proximity_reward = 0.1
        self.accurate_pass_reward = 0.2
        self.strategic_position_reward = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Enhance the base reward by adding strategic game-play rewards for scenarios focused on accurate passing, shooting accuracy, and strategic positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_proximity_reward": [0.0]*len(reward),
                      "accurate_pass_reward": [0.0]*len(reward),
                      "strategic_position_reward": [0.0] * len(reward)}

        # Check if observation is available
        if observation is None:
            return reward, components

        for i, single_obs in enumerate(observation):
            # Goal proximity reward
            if single_obs['ball_owned_team'] == 1:  # Check if the right team has the ball
                dist_to_opponent_goal = abs(single_obs['ball'][0] - 1)
                if dist_to_opponent_goal < 0.2:  # Ball is within scoring range
                    components["goal_proximity_reward"][i] = self.goal_proximity_reward
                    reward[i] += components["goal_proximity_reward"][i]

            # Accurate pass reward
            if single_obs['ball_owned_team'] == 1 and 'action' in single_obs:
                if single_obs['action'][6] or single_obs['action'][7]:  # Pass actions
                    components["accurate_pass_reward"][i] = self.accurate_pass_reward
                    reward[i] += components["accurate_pass_reward"][i]

            # Strategic positioning reward
            if single_obs['active'] == single_obs['designated']:  # If the active player is the same as the one making strategic decisions
                being_in_key_position = single_obs['right_team'][single_obs['active']][0] > 0.5  # In opponent's half
                if being_in_key_position:
                    components["strategic_position_reward"][i] = self.strategic_position_reward
                    reward[i] += components["strategic_position_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Check sticky actions state
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                if act:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
