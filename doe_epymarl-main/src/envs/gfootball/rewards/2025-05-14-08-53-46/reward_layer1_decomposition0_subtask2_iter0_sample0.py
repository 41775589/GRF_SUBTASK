import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds an additional reward for dribbling and approaching a shooting position,
    particularly focusing on ball control near the opponent's goal.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and any internal variables upon the start of a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current internal state of the reward wrapper, in case it's needed.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load the state previously saved for continuity in simulation or training.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Adjust the reward provided by the environment to incentivize dribbling near opponents 
        and maintaining control to move into scoring positions.
        """
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0],
            "positioning_reward": [0.0]
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Dribbling reward: check if the agent has the ball and is dribbling
            if o['ball_owned_player'] == o['active'] and o['sticky_actions'][9]:
                components["dribbling_reward"][rew_index] = 0.2

            # Positioning reward: encourage moving towards a good shooting position
            ball_x, ball_y = o['ball'][0], o['ball'][1]
            player_x, player_y = o['right_team'][o['active']][0], o['right_team'][o['active']][1]
            distance_to_goal = np.sqrt((1 - player_x)**2 + (0.44 - player_y)**2)

            if distance_to_goal < 0.3:  # closer to the goal in the opponent's half
                components["positioning_reward"][rew_index] = 0.3

            reward[rew_index] += components["dribbling_reward"][rew_index] + components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Steps through the environment using the given action and calculates rewards.
        """
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
