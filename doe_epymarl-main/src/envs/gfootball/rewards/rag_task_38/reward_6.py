import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to promote special behaviours:
    1. Initiating counterattacks through accurate long passes.
    2. Quick transitions from defense to attack.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transitioned_player_last_step = None
        self.long_pass_reward = 0.5
        self.transition_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transitioned_player_last_step = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.transitioned_player_last_step
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transitioned_player_last_step = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "long_pass_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_speed = np.linalg.norm(o['ball_direction'])
            distance_to_goal = o['right_team'][o['active']][0] - 1  # Assuming goal at x = 1
            
            # Reward for successful long passes
            if ball_speed > 0.1 and 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                components["long_pass_reward"][rew_index] = self.long_pass_reward
                reward[rew_index] += components["long_pass_reward"][rew_index]
            
            # Reward for quick transitions from defense to attack
            if self.transitioned_player_last_step is not None:
                if self.transitioned_player_last_step and 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                    # Check if the team has switched from not having possession to having possession 
                    # at a position closer to the opponent's goal
                    if any(pos[0] >= 0 and prev_pos[0] < 0 for pos, prev_pos in 
                           zip(o['right_team'], observation[self.transitioned_player_last_step]['right_team'])):
                        components["transition_reward"][rew_index] = self.transition_reward
                        reward[rew_index] += components["transition_reward"][rew_index]

            self.transitioned_player_last_step = rew_index
            
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
