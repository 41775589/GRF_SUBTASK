import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper designed to encourage and enhance mid to long-range passing effectiveness
    by rewarding agents based on precision and strategic use of passes in coordinated plays.
    """
    def __init__(self, env):
        super().__init__(env)
        self.passing_distance_threshold = 0.3  # Threshold for considering a pass long-range.
        self.passing_accuracy_bonus = 0.05  # Small bonus for every accurate long-range pass.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of collected checkpoints (if any) to the pickle object.
        """
        # No specific checkpoints to save in this particular reward wrapper
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore state from picked object.
        """
        # No specific checkpoints to restore in this particular reward wrapper
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        """
        Custom reward function that favors accurate long-range passing.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_passing_reward": [0.0, 0.0]
        }

        if observation is None:
            return reward, components
        
        for index in range(len(reward)):
            o = observation[index]
            # Check if we have a passing scenario
            if ('ball_owned_team' in o) and (o['ball_owned_team'] == (0 if index == 0 else 1)):
                # This agent's team owns the ball
                current_player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
                for other_player_pos in (o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']):
                    distance = np.linalg.norm(current_player_pos - other_player_pos)
                    # Reward if the pass was long enough and accurately reaches another team player
                    if distance > self.passing_distance_threshold:
                        components['precision_passing_reward'][index] += self.passing_accuracy_bonus
                        reward[index] += components['precision_passing_reward'][index]

        return reward, components

    def step(self, action):
        """
        Take an action using the base environment and modify the reward using the custom reward wrapper.
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
                if action:
                    self.sticky_actions_counter[i] += 1

        # Append sticky action counts to info for debugging purposes
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
            
        return observation, reward, done, info
