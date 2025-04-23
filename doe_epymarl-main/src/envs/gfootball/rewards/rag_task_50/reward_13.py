import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long passes between specific zones of the playfield."""

    def __init__(self, env):
        super().__init__(env)
        self.zone_checkpoints = [
            (-1, -0.33), (-1, 0.33),  # Left zones
            (0, -0.33), (0, 0.33),    # Middle zones
            (1, -0.33), (1, 0.33)     # Right zones
        ]
        self.pass_threshold = 0.5
        self.long_pass_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        new_ball_position = observation[0]['ball'][:2]  # get (x,y) of the ball

        if self.last_ball_position is not None:
            dist_moved = np.linalg.norm(new_ball_position - self.last_ball_position)
            # Check if the pass is long and crosses designed zones.
            if dist_moved > self.pass_threshold:
                # Check if moved through distinct zones
                start_zone = self.identify_zone(self.last_ball_position)
                end_zone = self.identify_zone(new_ball_position)
                if start_zone != end_zone and start_zone is not None and end_zone is not None:
                    for agent_index in range(len(reward)):
                        components["long_pass_reward"][agent_index] = self.long_pass_reward
                        reward[agent_index] += components["long_pass_reward"][agent_index]
        
        self.last_ball_position = new_ball_position

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

    def identify_zone(self, position):
        """Helper to identify which zone position belongs to."""
        x, y = position
        for idx, (zx, zy) in enumerate(self.zone_checkpoints):
            if abs(x - zx) <= 0.33 and abs(y - zy) <= 0.42:
                return idx
        return None
