import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for 'sweeper' role performance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.5
        self.tackle_reward = 0.7
        self.coverage_penalty = -0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = "state_info"
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "coverage_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Assign rewards based on defender/sweeper actions
            if o['game_mode'] in [3, 5]:  # Free Kick or Throw In
                if o['ball_owned_team'] == 0:  # ball_owned_team: 0 is the left team, 1 is the right team
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]

            # Tackle reward when the opposition has a goal kick but loses the ball
            if o['game_mode'] == 2 and o['ball_owned_team'] == 1:
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]

            # Check if the sweeper covers important areas when not having the ball
            if o['game_mode'] == 0 and o['ball_owned_team'] == -1:  # Normal play and no one owns the ball
                distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
                if distance_to_ball > 0.3:  # too far from the ball
                    components["coverage_penalty"][rew_index] = self.coverage_penalty
                    reward[rew_index] += components["coverage_penalty"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
