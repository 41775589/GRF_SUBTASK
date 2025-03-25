import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on enhancing counter-attack skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_possession = -1  # To track changes in ball possession.

    def reset(self):
        self.previous_ball_possession = -1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components
        components = {"base_score_reward": reward.copy(),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            current_obs = observation[rew_index]
            
            # Check for possession change from opponent to agent.
            if (self.previous_ball_possession != current_obs['ball_owned_team'] == 0):
                current_ball_pos = np.array(current_obs['ball'][:2])
                goal_position = np.array([1, 0])  # Position of the opponent's goal
                
                # Compute the Euclidean distance from current ball position to the opponent goal
                distance_to_goal = np.linalg.norm(current_ball_pos - goal_position)
                
                # As closer to opponent goal as better (inverted reward logic because closer means less distance)
                proximity_reward = 1 - min(distance_to_goal, 1)  # Normalize and invert
                components['counter_attack_reward'][rew_index] += proximity_reward
            
            # Update possession tracking
            self.previous_ball_possession = current_obs['ball_owned_team']

        # Total reward for this step
        reward = [
            base + counter for base, counter in zip(components["base_score_reward"], components["counter_attack_reward"])
        ]
        
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
