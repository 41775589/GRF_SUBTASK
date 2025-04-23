import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense, task-specific reward for mastering close-range attacks and dribbles against goalkeepers.
    Rewards precise shots and effective dribbling when close to the goal, especially in direct interaction with the goalkeeper.
    """
    def __init__(self, env):
        super().__init__(env)
        self.shot_precision_reward = 0.3
        self.dribble_effectiveness_reward = 0.2
        self.decision_making_reward = 0.5
        self.goalkeeper_interaction_distance_threshold = 0.1
        self.goal_location = np.array([1, 0])  # Assuming attacks towards right goal

        # Count for sticky actions to ensure quick decision making
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any state you need here from from_pickle
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        # Initialize reward components
        for rew_index, o in enumerate(observation):
            ball_position = np.array(o['ball'][:2])  # Only consider x, y coordinates
            goalkeeper_position = o['right_team'][o['right_team_roles'] == 0][0]  # Index of the goalkeeper in the right team

            # Calculate distance from the ball to the goalkeeper
            distance_to_goalkeeper = np.linalg.norm(ball_position - goalkeeper_position)

            if distance_to_goalkeeper < self.goalkeeper_interaction_distance_threshold:
                components["shot_precision_reward"] = self.shot_precision_reward
                components["dribble_effectiveness_reward"] = self.dribble_effectiveness_reward
                if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and 
                    'ball_owned_player' in o and o['ball_owned_player'] == o['active']):
                    reward[rew_index] += components["shot_precision_reward"] + components["dribble_effectiveness_reward"]
            
            # Reward for quick decision making
            if self.sticky_actions_counter.sum() < 3: # Assuming quick play involves fewer than 3 sticky actions
                components["decision_making_reward"] = self.decision_making_reward
                reward[rew_index] += components["decision_making_reward"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
fish Component and final reward values are added to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Count sticky actions for quick decision making component
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
