import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that promotes defensive behavior and quick transitions 
    into counterattacks by penalizing loss of ball possession and rewarding 
    good positioning and responsive actions.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset sticky actions counter on a new episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store the current state, including the sticky actions counter
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the sticky actions counter from saved state
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('CheckpointRewardWrapper', []))
        return from_pickle

    def reward(self, reward):
        # Access observation dictionary
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "possession_penalty": [0.0] * len(reward)}

        # Direct defensive maneuver rewards and quick transition rewards
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Increase reward for defensive positioning (distance from own goal)
            if 'left_team' in o and 'ball' in o:
                our_goal = np.array([-1, 0])  # Assuming playing on the left side
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                player_dist = np.linalg.norm(player_pos - our_goal)
                ball_dist = np.linalg.norm(ball_pos - our_goal)

                # Reward lower distance between player and our goal compared to the ball and our goal
                if player_dist < ball_dist:
                    components["positional_reward"][rew_index] = 0.1

            # Penalty for losing the ball possession
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # If right team owns the ball
                components["possession_penalty"][rew_index] = -0.2

            # Calculate final reward considering all components
            reward[rew_index] += components["positional_reward"][rew_index]
            reward[rew_index] += components["possession_penalty"][rew_index]

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
