import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards coordinated offensive plays between midfielders and strikers."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_advancement_reward = 0.05
        self.striker_finishing_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfielder_advancement_reward": [0.0] * len(reward),
                      "striker_finishing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][0]  # Using only x-coordinate for simplicity

            # Check if midfielders perform well in positioning and passing
            if 'right_team_roles' in o and o['right_team_roles'][o['active']] == 6:  # assuming role index 6 is Midfielder
                # Reward for advancing the ball toward the opponent's half
                if ball_pos > 0.3:
                    components["midfielder_advancement_reward"][rew_index] = self.midfielder_advancement_reward
                    reward[rew_index] += components["midfielder_advancement_reward"][rew_index]

            # Check if the striker secures ball close to the goal
            if 'right_team_roles' in o and o['right_team_roles'][o['active']] == 9:  # assuming role index 9 is Striker
                # Reward when a striker receives the ball close to the opponent's goal
                if ball_pos > 0.7 and o['ball_owned_team'] == 1:
                    components["striker_finishing_reward"][rew_index] = self.striker_finishing_reward
                    reward[rew_index] += components["striker_finishing_reward"][rew_index]

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
