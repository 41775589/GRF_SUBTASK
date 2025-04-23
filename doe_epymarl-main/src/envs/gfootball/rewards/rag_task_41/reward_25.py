import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward promoting attacking and creative play in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_approach_reward = 0.05
        self.creative_play_reward = 0.1
        self.ball_control_rewards = {}

    def reset(self):
        """Reset the sticky actions counter and the ball control rewards dictionary."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return state with added wrapper-specific state."""
        to_pickle['ball_control_rewards'] = self.ball_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from the previously saved state."""
        from_pickle = self.env.set_state(state)
        self.ball_control_rewards = from_pickle.get('ball_control_rewards', {})
        return from_pickle

    def reward(self, reward):
        """Modify the reward using customized logic promoting offensive play."""
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_approach_reward": [0.0] * len(reward),
                      "creative_play_reward": [0.0] * len(reward)}

        if observations is None:
            return reward, components

        assert len(reward) == len(observations)

        for index in range(len(reward)):
            obs = observations[index]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0 and \
               'ball_owned_player' in obs and obs['ball_owned_player'] == obs['active']:
                if index not in self.ball_control_rewards:
                    self.ball_control_rewards[index] = True
                    components["creative_play_reward"][index] += self.creative_play_reward
                    reward[index] += self.creative_play_reward

                # Calculate distance to opponent's goal:
                ball_pos = obs['ball']
                distance_to_goal = abs(ball_pos[0] - 1)
                component_reward = (1 - distance_to_goal) * self.goal_approach_reward
                components["goal_approach_reward"][index] += component_reward
                reward[index] += component_reward

        return reward, components

    def step(self, action):
        """Step function that also tracks custom rewards and components."""
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
