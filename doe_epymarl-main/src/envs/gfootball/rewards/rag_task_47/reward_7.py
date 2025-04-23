import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a new reward function focusing on sliding tackles during high-pressure defensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "sliding_tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'game_mode' in o and (o['game_mode'] == 3 or o['game_mode'] == 5) and o['ball_owned_team'] == -1:  # Free-kick or Throw-in mode
                # Reward a successful tackle
                if o['sticky_actions'][2] == 1:  # Sliding Tackle action
                    current_rewards = components["sliding_tackle_reward"][rew_index]
                    defense_pos = o['left_team'] if o['active'] in o['left_team'] else o['right_team']
                    ball_pos = o['ball'][0:2]  # consider x, y coordinates

                    # Prefer tackles closer to own goal to prevent dangerous plays
                    goal_pos = [1, 0] if o['active'] in o['left_team'] else [-1, 0]
                    distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(ball_pos))
                    tackle_effectiveness = max(0, 1 - distance_to_goal)  # More effective as it's closer to the goal

                    additional_reward = 0.1 * tackle_effectiveness
                    components["sliding_tackle_reward"][rew_index] += additional_reward
                    reward[rew_index] += components["sliding_tackle_reward"][rew_index]

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
            for i, active in enumerate(agent_obs['sticky_actions']):
                if active == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
