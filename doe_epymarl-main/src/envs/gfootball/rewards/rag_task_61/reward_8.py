import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function for enhancing team synergy during possession changes,
       emphasizing precise timing and strategic positioning during offensive and defensive moves."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner_team = -1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner_team = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'previous_ball_owner_team': self.previous_ball_owner_team}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner_team = from_pickle['CheckpointRewardWrapper']['previous_ball_owner_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            current_ball_owner = o['ball_owned_team']

            # Assign rewards based on possession changes emphasizing strategy and position
            if current_ball_owner != self.previous_ball_owner_team and current_ball_owner != -1:
                # Encourage change of possession by defensive action
                # Reward is higher if the change happens in strategic areas (e.g., near the own goal)
                if current_ball_owner != o['active']:
                    opponent_goal_dist = np.linalg.norm(
                        o['ball'] - np.array([1 if current_ball_owner == 1 else -1, 0]))
                    components["possession_change_reward"][i] = 1 - opponent_goal_dist
                    reward[i] += components["possession_change_reward"][i]

            self.previous_ball_owner_team = o['ball_owned_team']

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
