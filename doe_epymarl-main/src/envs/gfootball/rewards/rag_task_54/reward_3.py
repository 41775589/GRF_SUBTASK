import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focusing on enhancing collaborative plays
    between shooters and passers in order to exploit scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter = 0
        self.pass_completion_reward = 0.5
        self.goal_assist_reward = 1.0

    def reset(self):
        """
        Reset sticky actions counter and pass completion counter at the start of a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current state along with the pass completion count.
        """
        to_pickle['pass_completion_counter'] = self.pass_completion_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the environment state from a given pickled state, including pass completion count.
        """
        from_pickle = self.env.set_state(state)
        self.pass_completion_counter = from_pickle['pass_completion_counter']
        return from_pickle

    def reward(self, reward):
        """
        Enhance the reward mechanism with additional points for collaborative play
        between shooters and passers.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "goal_assist_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for pass completion
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                active_player_pos = o['left_team'][o['active']]
                teammates = o['left_team']
                for teammate_index, teammate_pos in enumerate(teammates):
                    if teammate_index != o['active']:
                        # Check if the ball has moved significantly towards another teammate
                        if np.linalg.norm(teammate_pos - active_player_pos) < np.linalg.norm(o['ball_direction']):
                            self.pass_completion_counter += 1
                            components["pass_completion_reward"][rew_index] += self.pass_completion_reward
                            reward[rew_index] += components["pass_completion_reward"][rew_index]

            # Additional reward for goal assists
            if o['score'][0] > 0:  # Assuming 0 index is scoring for the left team
                reward[rew_index] += self.goal_assist_reward
                components["goal_assist_reward"][rew_index] += self.goal_assist_reward

        return reward, components

    def step(self, action):
        """
        Take a step using the given action, calculate reward, and inject detailed
        components into the info dictionary returned by the env step.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Update info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = max(
                    self.sticky_actions_counter[i], action)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
