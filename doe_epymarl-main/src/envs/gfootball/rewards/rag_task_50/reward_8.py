import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing accurate long passes between specified zones of the pitch.
    Rewards are given based on how well agents execute long passes between different areas,
    focusing on vision, timing, and precision in ball distribution.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define the zones on the field for target long pass checkpoints
        self.pass_zones = [
            [0.0, 0.2],    # midfield central zone
            [-0.8, -1.0],  # left defensive zone
            [0.8, 1.0],    # right defensive zone
            [-0.8, 0.0],   # left offensive zone
            [0.8, 0.0]     # right offensive zone
        ]
        # Initialize sticky actions tracker
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Distance threshold for pass length to be considered 'long'
        self.long_pass_threshold = 0.5
        # Reward for executing a long pass correctly
        self.long_pass_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        # Additional states can be saved here if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Additional states can be loaded here if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if a pass is made by the controlled player
            if (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']):
                prev_ball_pos = o['ball'] - o['ball_direction']
                ball_pos = o['ball']
                start_zone = None
                end_zone = None

                # Find the zone of the starting position
                for zone_index, zone in enumerate(self.pass_zones):
                    if prev_ball_pos[0] >= zone[0] and prev_ball_pos[0] <= zone[1]:
                        start_zone = zone_index

                # Find the zone of the ending position
                for zone_index, zone in enumerate(self.pass_zones):
                    if ball_pos[0] >= zone[0] and ball_pos[0] <= zone[1]:
                        end_zone = zone_index

                # Check if the ball was passed from one zone to another and the pass is long enough
                if start_zone is not None and end_zone is not None and start_zone != end_zone:
                    dist = np.linalg.norm(ball_pos - prev_ball_pos)
                    if dist > self.long_pass_threshold:
                        components["long_pass_reward"][rew_index] = self.long_pass_reward
                        reward[rew_index] += self.long_pass_reward

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
