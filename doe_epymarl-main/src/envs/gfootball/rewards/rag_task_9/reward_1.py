import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for demonstrating offensive football skills such as passing,
    shooting, and dribbling to create scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize sticky_actions_counter as required
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define custom reward coefficients for various actions:
        # Complementing action efficiency and outcome in an offensive play.
        self.pass_coefficient = 0.15 # reward for successful passes
        self.shot_coefficient = 0.25  # reward for attempts on goal
        self.dribble_coefficient = 0.1 # reward for dribbles
        self.sprint_coefficient = 0.05 # reward for sprint actions
        self.goal_coefficient = 1.0   # reward for scoring a goal

    def reset(self):
        """
        Reset the environment and counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Serialize current state for saving game progress.
        """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state based on deserialized input.
        """
        return self.env.set_state(state)

    def reward(self, rewards):
        """
        Modifies the reward based on offensive skills being utilized effectively in gameplay.
        """
        observation = self.env.unwrapped.observation()
        player_actions = {"base_score_reward": rewards.copy(),
                          "pass_reward": [0.0] * len(rewards),
                          "shot_reward": [0.0] * len(rewards),
                          "dribble_reward": [0.0] * len(rewards),
                          "sprint_reward": [0.0] * len(rewards),
                          "goal_reward": [0.0] * len(rewards)}
        
        for idx in range(len(rewards)):
            obs = observation[idx]
            if obs is None:
                continue
            
            # Increment per action reward strategies which promote offensive play:
            # Reward passes and shots by looking at sticky actions increments.
            if obs['sticky_actions'][7] == 1:  # Pass (short or long)
                player_actions['pass_reward'][idx] += self.pass_coefficient
            
            if obs['sticky_actions'][9] == 1:  # Dribble 
                player_actions['dribble_reward'][idx] += self.dribble_coefficient
            
            if obs['sticky_actions'][8] == 1:  # Sprint
                player_actions['sprint_reward'][idx] += self.sprint_coefficient
            
            if obs['ball'][0] > 0.5 and rewards[idx] > 0:  # ball is near the opponent's goal
                player_actions['shot_reward'][idx] += self.shot_coefficient
            
            if rewards[idx] == 100:  # Goal event
                player_actions['goal_reward'][idx] += self.goal_coefficient
            
            # Sum up the modified rewards
            total_reward = (rewards[idx] +
                            player_actions['pass_reward'][idx] +
                            player_actions['shot_reward'][idx] +
                            player_actions['dribble_reward'][idx] +
                            player_actions['sprint_reward'][idx] +
                            player_actions['goal_reward'][idx])
                            
            rewards[idx] = total_reward

        return rewards, player_actions

    def step(self, action):
        """
        Environment step with extra information in returned dictionary.
        """
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
