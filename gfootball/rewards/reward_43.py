import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    Enhances the reward system by promoting offensive strategies like
    accurate shooting, effective dribbling, and strategic passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_accuracy_bonus = 0.5
        self.dribbling_evasion_bonus = 0.3
        self.passing_breakthrough_bonus = 0.4

    def reset(self):
        """ Resets the environment and any necessary internal data """
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Gets the state of the wrapper along with the environment's state """
        env_state = self.env.get_state(to_pickle)
        env_state['CheckpointRewardWrapper'] = {}
        return env_state

    def set_state(self, state):
        """
        Sets the state of the environment from the pickle,
        restoring the internal configuration of the wrapper.
        """
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Custom reward function to enhance learning of offensive strategies.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_bonus": [0.0] * len(reward),
            "dribbling_evasion_bonus": [0.0] * len(reward),
            "passing_breakthrough_bonus": [0.0] * len(reward)
        }

        if not observation:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            player_obs = observation[i]
            
            # Shooting accuracy: Bonus for shots close to goal
            if player_obs.get('ball_owned_player') == player_obs.get('active') and \
                    player_obs.get('game_mode') == 6:  # Assumes mode 6 is shot at goal
                components['shooting_accuracy_bonus'][i] = self.shooting_accuracy_bonus
                reward[i] += components['shooting_accuracy_bonus'][i]

            # Dribbling evasion: Bonus for maintaining possession under pressure
            if player_obs.get('ball_owned_player') == player_obs.get('active') and \
                    np.any(player_obs.get('sticky_actions')[8:]):  # Assumes indices 8-9 are dribble actions
                components['dribbling_evasion_bonus'][i] = self.dribbling_evasion_bonus
                reward[i] += components['dribbling_evasion_bonus'][i]

            # Passing breakthrough: Bonus for successful passes breaking opponent lines
            if player_obs.get('game_mode') == 2:  # Assumes mode 2 is successful long/high pass
                components['passing_breakthrough_bonus'][i] = self.passing_breakthrough_bonus
                reward[i] += components['passing_breakthrough_bonus'][i]

        return reward, components

    def step(self, action):
        """
        Step function to execute actions in the environment
        and modify the rewards using the reward() method.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
