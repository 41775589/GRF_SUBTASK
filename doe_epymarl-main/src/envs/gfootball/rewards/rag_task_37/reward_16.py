import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for advanced ball control and passing
    under pressure. It encourages mastering Short Pass, High Pass, and Long Pass
    during tight game situations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, obs in enumerate(observation):
            # Initialize reward components
            pass_control_reward = 0.0

            if 'ball_owned_team' in obs and obs['ball_owned_team'] == obs['team']:
                active_player_has_ball = (obs['ball_owned_player'] == obs['active'])
                tight_situation = np.any(obs['sticky_actions'][5:7])  # Checking if High Pass or Long Pass actions are active.

                # Reward for successful passes under pressure.
                if tight_situation and active_player_has_ball:
                    ball_speed = np.linalg.norm(obs['ball_direction'][:2])
                    if ball_speed > 0.01:  # Assuming a pass if the ball speed is significant
                        pass_control_reward += 0.2

            # Add components and accumulate total reward
            components["passing_reward"][rew_index] = pass_control_reward
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
