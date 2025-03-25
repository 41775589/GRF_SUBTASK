import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for sprinting while dribbling towards tight defensive lines to simulate evasion and
    ball control in offensive positions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for sticky actions used by agents

    def reset(self):
        """
        Reset the wrapper state along with the environment's state.
        """
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the wrapper along with the environment's state.
        """
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the wrapper from the unpickled state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on advanced dribbling and sprinting actions.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_sprint_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            dribbling = o['sticky_actions'][9]
            sprinting = o['sticky_actions'][8]
            ball_control = (o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'])

            if dribbling and sprinting and ball_control:
                reward_increase = 0.05  # Increase reward if dribbling and sprinting with ball control
                components["dribble_sprint_reward"][rew_index] = reward_increase
                reward[rew_index] += reward_increase

        return reward, components

    def step(self, action):
        """
        Execute a step using the given action, calculate and apply reward modifications.
        """
        observation, reward, done, info = self.env.step(action)

        # Apply the reward wrapper logic
        reward, components = self.reward(reward)

        # Update info dictionary with reward components for transparency in evaluating reward function
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        # Reset the sticky actions counter 
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
