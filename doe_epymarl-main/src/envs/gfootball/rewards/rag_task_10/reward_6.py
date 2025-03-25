import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focused on defensive actions such as
    intercepting passes, marking players, and preventing goals.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._interception_reward = 0.7
        self._defense_marking_reward = 0.3
        self._prevented_goal_reward = 2.0

    def reset(self):
        """
        Resets the environment and clears sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the environment.
        """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment.
        """
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Reward the agent for defensive actions.
        """
        observation = self.env.unwrapped.observation()
        base_reward = np.array(reward)
        defensive_components = np.zeros_like(reward)

        if observation is None:
            return base_reward, {'base_reward': base_reward, 'defensive_reward': defensive_components}

        for i, o in enumerate(observation):
            game_mode = o['game_mode']
            ball_owned_team = o['ball_owned_team']
            
            # Encourage interception:
            if game_mode == 3:  # Assuming mode 3 is close to opponent's action phase
                if ball_owned_team == 1:  # Ball owned by opponent
                    defensive_components[i] += self._interception_reward

            # Encourage marking in non-ball possession mode
            if ball_owned_team == 1 or ball_owned_team == -1:
                defensive_components[i] += self._defense_marking_reward

            # Bonus for preventing a goal
            if game_mode != 0 and ball_owned_team == 1:
                defensive_components[i] += self._prevented_goal_reward
            
        final_rewards = base_reward + defensive_components
        return final_rewards, {'base_reward': base_reward, 'defensive_reward': defensive_components}

    def step(self, action):
        """
        Step the environment with the given actions and calculate rewards.
        """
        ob, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info['final_reward'] = np.sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        return ob, new_reward, done, info
