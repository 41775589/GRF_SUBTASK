import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive actions reward."""

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
        # Restore any additional state stored in this wrapper.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # If the opponent has the ball
            if o['ball_owned_team'] == 1:
                # Compute proximity of agent to ball
                player_pos = o['right_team'][o['active']]
                ball_pos = o['ball'][:2]
                dist_to_ball = np.linalg.norm(player_pos - ball_pos)

                # Reward defensive actions based on closeness to the ball and ball direction towards own goal
                if dist_to_ball < 0.1 and np.dot(o['ball_direction'][:2], np.array([1, 0])) > 0:
                    components["defensive_reward"][rew_index] = 0.5  # arbitrary positive reward for getting close to the ball

                # Additional reward for tackling or intercepting
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming 9 is index for tackling
                    components["defensive_reward"][rew_index] += 0.3  # extra reward for executing a defensive action

                # Update composite reward
                reward[rew_index] += components["defensive_reward"][rew_index]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
