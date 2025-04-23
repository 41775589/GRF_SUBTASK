import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards accurate long passes between specified zones
    in the playfield, enhancing the agent's ability to execute passes
    that connect different areas with precision and timing.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones as coordinates (left_x, right_x, bottom_y, top_y)
        self.pass_zones = [
            (-1.0, -0.5, -0.42, 0.42),  # Left side of the field
            (-0.5, 0.0, -0.42, 0.42),   # Middle-left 
            (0.0, 0.5, -0.42, 0.42),    # Middle-right
            (0.5, 1.0, -0.42, 0.42)     # Right side of the field
        ]
        self.last_zone = None
        self.pass_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_zone = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckPointRewardWrapper'] = {
            'last_zone': self.last_zone
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_zone = from_pickle['CheckPointRewardWrapper']['last_zone']
        return from_pickle

    def zone_check(self, x, y):
        for i, (lx, rx, by, ty) in enumerate(self.pass_zones):
            if lx <= x <= rx and by <= y <= ty:
                return i
        return None

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "passing_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Evaluate long pass reward
        for rew_index, o in enumerate(observation):
            # Check if the ball is owned by any team
            if o['ball_owned_team'] in [-1, 0]:
                continue

            # Calculate current ball zone
            ball_x, ball_y = o['ball'][:2]
            current_zone = self.zone_check(ball_x, ball_y)

            # Check if a pass has been made to a different zone
            if self.last_zone is not None and current_zone is not None:
                distance_travelled = abs(current_zone - self.last_zone)
                if distance_travelled > 1:
                    components["passing_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

            self.last_zone = current_zone

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
