import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on efficient passing under defensive pressure."""

    def __init__(self, env):
        super().__init__(env)
        # Number of checkpoints to reward successful passes
        self.num_checkpoints = 5
        self.checkpoint_reward = 0.2
        self.passed_checkpoints = set()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """Reset the environment and clear checkpoints."""
        self.passed_checkpoints.clear()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state with checkpoints."""
        to_pickle['passed_checkpoints'] = self.passed_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state including checkpoints."""
        from_pickle = self.env.set_state(state)
        self.passed_checkpoints = from_pickle['passed_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            # Defensive pressure is higher near the opponent's goal
            if o['game_mode'] in (1, 2, 4, 5): # Non-regular modes where passing is crucial
                continue
            
            # Determining if the ball is with the active agent
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if o['active'] not in self.passed_checkpoints:
                    reward[idx] += self.checkpoint_reward
                    components["passing_reward"][idx] = self.checkpoint_reward
                    self.passed_checkpoints.add(o['active'])
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
