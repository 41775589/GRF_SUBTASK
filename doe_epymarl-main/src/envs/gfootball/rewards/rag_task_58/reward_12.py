import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive coordination and transition from defense to 
    attack reward to the standard game environment."""

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
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defense_transition_reward": [0.0]*len(reward),
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if it's defending scenario
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                player_pos = o['left_team'][o['active']]
                # Let's say we get small rewards for maintaining high distance
                # from the goal and towards ball possession as a means of defense
                goal_distance = np.sqrt((player_pos[0] - (-1))**2 + (player_pos[1])**2)
                ball_distance = np.sqrt((o['ball'][0] - player_pos[0])**2 + (o['ball'][1] - player_pos[1])**2)

                # Adjust these coefficients based on importance
                defense_reward = (goal_distance * 0.2) + (0.5 / (ball_distance + 0.1))
                components["defense_transition_reward"][rew_index] = defense_reward
                reward[rew_index] += defense_reward
            
            # Check if it's transitioning to attack
            elif o['ball_owned_team'] == 0:  # Our team has the ball
                player_pos = o['left_team'][o['active']]
                goal_distance = np.sqrt(((player_pos[0] - 1))**2 + (player_pos[1] - 0)**2)
                
                # We can add transitioning reward based on how forward player is moving towards the opponent’s goal
                transitioning_reward = (1 - goal_distance) * 0.7
                components["defense_transition_reward"][rew_index] += transitioning_reward
                reward[rew_index] += transitioning_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Record the sticky actions' activation count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
