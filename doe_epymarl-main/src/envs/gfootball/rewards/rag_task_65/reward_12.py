import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances offensive play development by focusing on passing and shooting in game-like contexts."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_accuracy_threshold = 0.1  # Distance threshold to consider a pass successful
        self.goal_shot_reward = 1.0  # Reward for shooting towards the goal
        self.pass_reward = 0.5  # Reward for a successful pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and clear the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State getter that can be extended with the wrapper's specific state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State setter that retrieves the wrapper's specific state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards given the strategic context."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_shot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goal_dist = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
            if goal_dist < 0.2 and o['active'] == o['ball_owned_player']:
                components["goal_shot_reward"][rew_index] = self.goal_shot_reward
                reward[rew_index] += components["goal_shot_reward"][rew_index]

            # Evaluate pass quality
            if 'right_team' in o and o['active'] in o['right_team']:
                teammates = [p for idx, p in enumerate(o['right_team']) if idx != o['active']]
                pass_distances = [np.linalg.norm(o['right_team'][o['active']] - tm) for tm in teammates]
                close_passes = [d for d in pass_distances if d < self.pass_accuracy_threshold]
                if close_passes:
                    components["pass_reward"][rew_index] = self.pass_reward * len(close_passes)
                    reward[rew_index] += components["pass_reward"][rew_index]

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
                if action:
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
