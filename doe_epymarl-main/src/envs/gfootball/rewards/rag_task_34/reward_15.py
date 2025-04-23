import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for dribble and precise close-range shot control against goalkeepers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.shot_precision_threshold = 0.1  # The proximity to goal to consider a shot as 'close-range'
        self.dribble_efficiency = 0.2  # Encourages maintaining control when close to the opponent's goal
        self.decisiveness_bonus = 0.5  # Reward for quick decisive actions near the goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracks the number of sticky actions

    def reset(self):
        """Reset the game environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the wrapper's state into a pickleable object."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve the state from a pickled object."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards returned by the environment based on task-specific conditions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_precision": 0.0, "dribble_efficiency": 0.0, "decisiveness_bonus": 0.0}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            ball_pos = o['ball'][:2]  # only consider x, y
            goal_pos = np.array([1, 0])  # right goal position

            # Calculate distance to goal
            distance_to_goal = np.linalg.norm(ball_pos - goal_pos)

            # Check for closeness to goal and possession
            if distance_to_goal <= self.shot_precision_threshold and o['ball_owned_team'] == 1:
                components["shot_precision"] = self.decisiveness_bonus
                reward[i] += components["shot_precision"]

            # Assess dribbling efficiency near the opposition's goal
            if o['ball_owned_team'] == 1 and (o['sticky_actions'][9] == 1):  # Assuming 9 is dribble action
                components["dribble_efficiency"] = self.dribble_efficiency
                reward[i] += components["dribble_efficiency"]

            # Reward quick, effective plays near the goal
            if distance_to_goal <= self.shot_precision_threshold * 2 and o['steps_left'] > 2950:
                components["decisiveness_bonus"] = self.decisiveness_bonus
                reward[i] += components["decisiveness_bonus"]

        return reward, components

    def step(self, action):
        """Execute a step in the environment, modify the rewards, and return observations and other info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
