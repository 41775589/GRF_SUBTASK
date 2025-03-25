import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_zones = 5  # Divide the x-axis of the opponent's half into 5 zones for reward checkpoints
        self.checkpoint_value = 0.2
        self.reward_state = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_state = [0] * self.num_zones
        return self.env.reset()

    def reward(self, reward):
        base_reward = reward  # Original game reward
        extra_reward = [0.0] * len(reward)  # Initialize extra rewards

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        for i, obs in enumerate(observation):
            ball_x, ball_y = obs['ball'][0], obs['ball'][1]
            ball_owned_team = obs.get('ball_owned_team', -1)

            if ball_owned_team == 1:  # If the right team has the ball
                # Only give rewards for strategic placement and movement if our agent has the ball
                if ball_x > 0:  # Ball is on the opponent's side
                    zone_idx = min(int((ball_x + 1) // (2 / self.num_zones)), self.num_zones - 1)
                    if self.reward_state[zone_idx] == 0:
                        extra_reward[i] += self.checkpoint_value
                        self.reward_state[zone_idx] = 1
        
        enhanced_rewards = [br + er for br, er in zip(base_reward, extra_reward)]
        return enhanced_rewards

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'reward_state': self.reward_state}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.reward_state = from_pickle['CheckpointRewardWrapper']['reward_state']
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        enhanced_rewards = self.reward(reward)
        info["final_reward"] = sum(enhanced_rewards)
        for key, er in enumerate(enhanced_rewards):
            info[f"reward_component_{key}"] = er
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, enhanced_rewards, done, info
