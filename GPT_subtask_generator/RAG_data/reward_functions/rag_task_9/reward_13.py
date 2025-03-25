import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive football actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
        components = {"base_score_reward": reward.copy()}

        for i in range(len(reward)):
            o = observation[i]

            # Initialize reward components for each agent
            components.setdefault(f'action_for_{i}', [0.0] * len(reward))

            # Check if agent has the possession of the ball
            if o.get('ball_owned_team') == 1 and o.get('ball_owned_player') == o.get('active'):
                # Positive reward for dribbling towards the opponent's goal
                if 'action_dribble' in o['sticky_actions']:
                    components[f'action_for_{i}'][i] += 0.1

                # Positive reward for successful passes and shots directed towards the goal
                if o['game_mode'] == 0:  # Normal game mode
                    if o['sticky_actions'][2] or o['sticky_actions'][1]:  # Action for Shot or Long Pass
                        components[f'action_for_{i}'][i] += 0.2

                    if o['sticky_actions'][0]:  # Action for Short Pass
                        components[f'action_for_{i}'][i] += 0.1
                
                # Sprint action, promoting faster play
                if o['sticky_actions'][8]:  # Action for Sprint
                    components[f'action_for_{i}'][i] += 0.05

            # Update the reward from components
            reward[i] += sum(components[f'action_for_{i}'])
        
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
