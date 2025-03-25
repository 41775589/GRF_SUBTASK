import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering long passes and precision in football.
    The reward increases for successful long passes and maintaining possession during those passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_distance_threshold = 0.5  # Threshold to consider a pass as long
        self.prev_ball_position = np.zeros(3)
        self.pass_bonus = 0.2  # Reward for successful long pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['prev_ball_position'] = self.prev_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_ball_position = from_pickle['prev_ball_position']
        return from_pickle

    def reward(self, reward):
        """
        Augments the reward by providing a bonus for successful long passes.
        A long pass is defined by the movement of the ball over a predetermined distance threshold
        in one step, without changing ball possession.
        """
        current_ball_position = self.env.unwrapped.observation()[0]['ball']
        distance_moved = np.linalg.norm(current_ball_position - self.prev_ball_position)

        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": 0.0}

        # Check if the ball moved a long distance
        if distance_moved >= self.pass_distance_threshold and self.env.unwrapped.observation()[0]['ball_owned_team'] != -1:
            components["pass_bonus"] = self.pass_bonus
            reward += components["pass_bonus"]

        self.prev_ball_position = current_ball_position
        
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
