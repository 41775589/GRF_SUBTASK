import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for long, accurate passes between specific playfield areas.
    This aims to promote vision, timing, and precision in ball distribution.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define checkpoints that represent targeted regions for long passes
        self.checkpoints = [
            # Coordinates are (x, y) of the center of each region
            (-0.6, 0),   # Left midfield
            (0, 0.3),    # Center top midfield
            (0, -0.3),   # Center bottom midfield
            (0.6, 0),    # Right midfield
        ]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_threshold = 0.6  # Breadth of acceptable pass completion area
        self.pass_distance_threshold = 0.5  # Minimum distance to count as a "long pass"
        self.reward_per_pass = 0.3  # Reward added for each qualifying pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, (o, rew) in enumerate(zip(observation, reward)):
            if o['ball_owned_team'] == 1:  # Ensure the team owning the ball is right team
                current_player = o['right_team'][o['active']]
                pass_target = o['ball'] if o['ball_owned_player'] == -1 else None

                if pass_target is not None:
                    for checkpoint in self.checkpoints:
                        cp_x, cp_y = checkpoint
                        if (np.linalg.norm(pass_target - current_player) > self.pass_distance_threshold and
                                np.linalg.norm(pass_target - np.array([cp_x, cp_y])) < self.pass_quality_threshold):
                            components['pass_reward'][i] += self.reward_per_pass
                            break  # Consider only one rewarded pass per step

            rew += components['pass_reward'][i]
            reward[i] = rew
        
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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
