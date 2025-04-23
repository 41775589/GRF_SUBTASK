import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for executing accurate long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.3  # distance to consider a pass long.
        self.accuracy_factor = 0.1  # coefficient for rewarding accuracy.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle: dict) -> dict:
        """Get the current state with additional information."""
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state: dict):
        """Set the environment state with wrapped specific states."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward: list[float]) -> tuple[list[float], dict[str, list[float]]]:
        """Reward function specialized in rewarding accurate long passes."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward),
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Process each agent's observation and rewards.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # current team has the ball.
                if 'ball_owned_player' in o and o['active'] == o['ball_owned_player']:  # active player possesses the ball.
                    distance_travelled = np.linalg.norm(o['ball_direction'][:2])
                    if distance_travelled > self.pass_threshold:
                        # Reward directly proportional to the distance and the accuracy consideration.
                        components["long_pass_reward"][rew_index] = self.accuracy_factor * distance_travelled
                        reward[rew_index] += components["long_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Environment step wrapped with our reward modification logic."""
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
