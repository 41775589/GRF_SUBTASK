import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances training for scenario-based shooting and passing skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_threshold = 0.5  # Arbitrary threshold for incentivizing good passes
        self.shooting_bonus = 0.1     # Bonus for shooting close to the goal
        self.positioning_bonus_multiplier = 0.05  # Reward for strategic player positioning

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, from_pickle):
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10))
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'passing_bonus': [0.0] * len(reward),
            'shooting_bonus': [0.0] * len(reward),
            'positioning_bonus': [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Add custom reward components based on the specifics of the scenario
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 1:  # example for the right team
                # Calculate distance to opponent goal (goal post at x=1)
                distance_to_goal = 1 - o['ball'][0]
                if distance_to_goal < 0.2:  # Close range shooting
                    components['shooting_bonus'][i] = self.shooting_bonus * (0.2 - distance_to_goal)
            
                # Encourage passing by checking if controlled players are positioned to receive
                teammate_positions = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
                for teammate in teammate_positions:
                    if np.linalg.norm(teammate - o['ball'][:2]) < self.passing_threshold:
                        components['passing_bonus'][i] += 0.01  # small reward for feasible passing options

                # Bonus for strategic positioning (e.g., moving to open spaces)
                components['positioning_bonus'][i] = self.positioning_bonus_multiplier * (1 - np.max(np.abs(teammate_positions)))

            reward[i] += sum(components[k][i] for k in components if k != 'base_score_reward')

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
