import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes strategic positioning and transitioning between defensive and offensive plays."""

    def __init__(self, env):
        super().__init__(env)
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
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] != 0: # Non-normal gameplay (e.g. Kickoff, Corner)
                continue
            
            # Calculate rewards for maintaining strategic positions:
            # Defensive when the ball is near their goal, offensive when near the opponent's goal.
            own_goal_distance = np.linalg.norm(o['ball'] - [-1, 0])
            opponent_goal_distance = np.linalg.norm(o['ball'] - [1, 0])

            pos_reward = 0.0
            if own_goal_distance < 0.3:  # Defensive play
                pos_reward += 0.1 * (1 - own_goal_distance)  # Higher reward closer to own goal
            if opponent_goal_distance < 0.3:  # Offensive play
                pos_reward += 0.1 * (1 - opponent_goal_distance)  # Higher reward closer to opponent goal

            components["positional_reward"][rew_index] = pos_reward
            reward[rew_index] += components["positional_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions in the observation
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
