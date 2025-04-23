import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for the coordination between midfielders and strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # Initialize components to store different reward parts
        components = {'base_score_reward': reward.copy(),
                      'midfielder_reward': [0.0] * len(reward),
                      'striker_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        # Scan through all observations and apply midfielders-strikers coordination logic
        for i, o in enumerate(observation):
            if 'right_team_roles' in o and 'ball_owned_player' in o:
                # Reward the coordination between midfielders and strikers
                if o['ball_owned_player'] >= 0:
                    player_role = o['right_team_roles'][o['ball_owned_player']]
                    if player_role in {4, 6, 8}:  # Midfielder roles
                        # Reward for midfielders exploring and creating space
                        components['midfielder_reward'][i] += 0.1
                    elif player_role in {9}:  # Striker roles
                        # Reward for strikers finishing the play
                        components['striker_reward'][i] += 0.2
                
            # Calculate total reward for each agent
            reward[i] = (1 * components['base_score_reward'][i] +
                         1.5 * components['midfielder_reward'][i] +
                         2 * components['striker_reward'][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
