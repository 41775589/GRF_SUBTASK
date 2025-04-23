import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic positioning and movement transitions in gameplay."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for positioning and movement transitions
        self.positioning_weight = 0.05
        self.movement_transition_weight = 0.1
        self.prev_player_distances = None
        self.positioning_threshold = 0.2  # Distance threshold to opponent's goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_player_distances = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        state['prev_player_distances'] = self.prev_player_distances
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = state['sticky_actions_counter']
        self.prev_player_distances = state['prev_player_distances']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        rew_adjustment = [0.0] * len(reward)  

        for i, o in enumerate(observation):
            # Calculate positioning to opponent goal
            goal_y = 0.0
            goal_x = 1.0 if o['ball_owned_team'] == 0 else -1.0
            distance_to_goal = np.sqrt((o['ball'][0] - goal_x)**2 + (o['ball'][1] - goal_y)**2)
            
            # Create a positioning reward for getting close to the strategic position
            if distance_to_goal < self.positioning_threshold:
                positioning_reward = self.positioning_weight
            else:
                positioning_reward = 0
            
            # Calculate movement transition reward based on changes in distance
            if self.prev_player_distances is not None:
                change_in_distance = self.prev_player_distances[i] - distance_to_goal
                movement_transition_reward = self.movement_transition_weight * max(0, change_in_distance)
            else:
                movement_transition_reward = 0

            rew_adjustment[i] += positioning_reward + movement_transition_reward

        # Update previous distances
        self.prev_player_distances = [np.sqrt((o['ball'][0] - goal_x)**2 + (o['ball'][1] - goal_y)**2) for o in observation]
        
        # Apply adjustments
        for i in range(len(reward)):
            reward[i] += rew_adjustment[i]
        
        components["strategic_positioning"] = rew_adjustment
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
                self.sticky_actions_counter[i] += action
                
        return observation, reward, done, info
