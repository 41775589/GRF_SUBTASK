import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint based reward to encourage defending strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initializes sticky actions counter

    def reset(self):
        """Reset's environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get checkpoint states for saving."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore checkpoint states from loaded state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modify reward based on defending-oriented tasks."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "stop_move_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Reward for successful tackle
            if o['game_mode'] == 7 and o['ball_owned_team'] == 0:  # assuming game mode 7 indicates a tackle
                components["tackle_reward"][rew_index] += 0.2  # reward increment for tackling

            # Reward for stopping movement efficiently near an opponent
            distance = np.min(np.linalg.norm(o['left_team'] - o['ball'], axis=1))
            if distance < 0.1:  # if within 0.1 distance unit from the ball
                components["stop_move_reward"][rew_index] += 0.1

            # Reward for successful passing under pressure
            if o['game_mode'] == 5 and o['ball_owned_team'] == 0:  # Assuming game mode 5 is a pressured passing scenario
                components["pressure_pass_reward"][rew_index] += 0.3

            # Summing up the additional rewards with existing game reward
            reward[rew_index] += sum(components[c][rew_index] for c in components)

        return reward, components

    def step(self, action):
        """Execute environment step and modify the rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action_state in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action_state
        
        return observation, reward, done, info
