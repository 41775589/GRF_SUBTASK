import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective short passing under defensive pressure, focusing on ball retention and successful distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # Only modify reward when our team owns the ball
                ball_controlled = (o['ball_owned_player'] == o['active'])
                player_is_defended = np.any([np.linalg.norm(o['right_team'][i] - o['left_team'][o['active']]) < 0.1 for i in range(len(o['right_team']))])

                if ball_controlled:
                    components.setdefault("passing_reward", [0.0] * len(reward))
                    components["passing_reward"][rew_index] = 0.2  # Reward for controlling the ball
                    if o['sticky_actions'][9]:  # Check if 'action_dribble' is used
                        components["passing_reward"][rew_index] += 0.3
                    if player_is_defended:
                        components["passing_reward"][rew_index] += 0.5  # Extra reward for retaining the ball under pressure
                    
                    reward[rew_index] += components["passing_reward"][rew_index]

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
