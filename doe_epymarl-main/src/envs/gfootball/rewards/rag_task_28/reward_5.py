import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for dribbling skills against the goalkeeper.
    Encourages quick feints, sudden direction changes, and maintaining ball control under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and the sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment to serialize.
        """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment from deserialization.
        """
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Applies a custom reward function to encourage dribbling against the goalkeeper.
        
        Args:
            reward: The original reward from the environment.
        
        Returns:
            A tuple of the updated reward and a dictionary of components contributing to the reward.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_training_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Encourage dribbling: proximity to goalkeeper in possession of the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Goalkeeper proximity reward
                goalie_pos = o['right_team'][0]  # assuming index 0 is the goalkeeper
                player_pos = o['left_team'][o['designated']]
                distance_to_goalie = np.sqrt((player_pos[0] - goalie_pos[0])**2 + (player_pos[1] - goalie_pos[1])**2)
                
                # Reward for being close to the goalkeeper and having the ball
                if distance_to_goalie < 0.1:
                    components['dribbling_training_reward'][rew_index] = 1.0  # Strong reward for close dribbling
                elif distance_to_goalie < 0.2:
                    components['dribbling_training_reward'][rew_index] = 0.5  # Less reward for less closeness
                    
                # Updating the reward
                reward[rew_index] += components['dribbling_training_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        """
        Take an action computed previously by an algorithm.
        
        Args:
            action: The action to be taken.
        
        Returns:
            obs, reward, done, info: Observation, adjusted reward, environment flag to end, and supporting info.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Include each component for debugging and learning insights
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update for sticky actions, the number of steps when they are applied.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            for i, action_presence in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_presence

        return observation, reward, done, info
