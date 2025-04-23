import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that incentivizes agents to effectively clear the ball from defensive zones under pressure,
    by providing additional rewards for clearing distance and safety (no immediate opponent possession).
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define regions in the defensive half where clearing is crucial
        self.defensive_zone_threshold = -0.2  # Defensive zone threshold on x-axis
        self.clearing_reward = 0.3            # Reward for clearing the ball beyond this region

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        return from_picle

    def reward(self, reward):
        # Capture the base environment reward
        components = {"base_score_reward": reward.copy()}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Encourage clearing the ball by the actively controlled players
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                ball_pos = o['ball'][0]  # Extract x-position of the ball
                if ball_pos < self.defensive_zone_threshold:
                    # Check if ball is cleared out of defensive zone
                    next_ball_pos = ball_pos + o['ball_direction'][0]
                    if next_ball_pos > self.defensive_zone_threshold:
                        reward[idx] += self.clearing_reward

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
