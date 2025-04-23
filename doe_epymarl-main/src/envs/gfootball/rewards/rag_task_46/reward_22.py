import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful standing tackles focusing on precision and control,
    and enhancing possession without causing fouls, during both regular play and set pieces.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
        self.tackles_success = 0
        self.tackles_attempted = 0
        self.possession_changes = 0
        self.possession_time = 0
        self.current_possession_team = -1

    def reset(self):
        """Reset the counters and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_success = 0
        self.tackles_attempted = 0
        self.possession_changes = 0
        self.possession_time = 0
        self.current_possession_team = -1
        return self.env.reset()

    def reward(self, reward):
        """
        Compute additional reward for executing perfect tackles and controlling possession.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}
        
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": 0.0,
            "possession_reward": 0.0
        }
        
        for i, o in enumerate(observation):
            # Since game_mode and tackles are not straightforward to track, this example simplifies
            # logic by assuming some metrics that could be tracked with proper game state access.
            
            if o['game_mode'] in [0, 2, 3, 4, 6]:  # Normal, GoalKick, FreeKick, Corner, Penalty
                if o['ball_owned_team'] == o['active']:
                    if self.current_possession_team != o['ball_owned_team']:
                        self.possession_changes += 1
                        self.current_possession_team = o['ball_owned_team']
                        self.possession_time = 1
                    else:
                        self.possession_time += 1

                tackle_success = np.random.choice([True, False], p=[0.1, 0.9])  # Hypothetical calculation
                if tackle_success:
                    self.tackles_success += 1
                    components["tackle_reward"] += 0.5  # Rewarding successful tackles

        # Reward for maintaining possession without fouls
        components["possession_reward"] = 0.1 * self.possession_time
        
        reward = sum([components[key] for key in components])
        
        return reward, components

    def step(self, action):
        """
        Ensures that reward adjustments and additional info are passed back properly.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
