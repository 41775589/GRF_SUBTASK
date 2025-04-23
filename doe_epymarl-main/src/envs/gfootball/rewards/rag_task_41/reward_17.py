import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward encouraging attacking plays,
    focusing on finishing and offensive creativity while handling match-like pressures
    and defensive setups.
    """

    def __init__(self, env):
        super().__init__(env)
        self.offensive_checkpoints = {}
        self.num_forward_checkpoints = 5
        self.checkpoint_reward_amount = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['offensive_checkpoints'] = self.offensive_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.offensive_checkpoints = from_pickle['offensive_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, copy=True),
                      "attack_reward": np.zeros(len(reward))}
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            # Encourage forward movement of the ball by the player
            ball_position = o['ball'][0]  # Only consider x-coordinate for simplicity

            # Calculate checkpoints dynamically based on ball position towards opponent goal
            checkpoint_each = 1 / self.num_forward_checkpoints
            ball_progress = (ball_position + 1) / 2  # Normalize to range [0, 1] from [-1, 1]
            current_checkpoint = int(ball_progress // checkpoint_each)
            
            # Only reward if this is a new furthest checkpoint reached without loss of ball
            if o['ball_owned_team'] == 0:  # Assuming playing left to right
                max_checkpoint = self.offensive_checkpoints.get(i, -1)
                if current_checkpoint > max_checkpoint:
                    self.offensive_checkpoints[i] = current_checkpoint
                    components["attack_reward"][i] = self.checkpoint_reward_amount
                    reward[i] += components["attack_reward"][i]
        
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
