import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for promoting attacking skills in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_position = None

    def reset(self):
        """Reset the environment for a new episode."""
        self.previous_position = None
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to be serialized."""
        to_pickle['CheckpointRewardWrapper'] = {'previous_position': self.previous_position}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from serialization."""
        from_pickle = self.env.set_state(state)
        self.previous_position = from_pickle['CheckpointRewardWrapper']['previous_position']
        return from_pickle

    def reward(self, reward):
        """Calculate the reward, adding a component for attacking skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "attacking_enhancement_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                ball_position = np.array(o['ball'][:2])
                if self.previous_position is not None:
                    movement_towards_goal = ball_position[0] - self.previous_position[0]
                    # Reward positive movements towards the opponent's goal
                    if movement_towards_goal > 0:
                        components["attacking_enhancement_reward"][rew_index] = movement_towards_goal * 0.1
                self.previous_position = ball_position

            reward[rew_index] += components["attacking_enhancement_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Environment step, encapsulates environment's step and applies reward modifications."""
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
