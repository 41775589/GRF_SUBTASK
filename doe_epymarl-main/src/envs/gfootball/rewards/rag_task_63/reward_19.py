import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to train a goalkeeper in tasks such as shot stopping, 
    quick decision-making for ball distribution under pressure, 
    and effective communication with defenders.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment for saving.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment for loading.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Augment the reward based on goalkeeper-specific tasks: shot stopping, decision-making, and communication.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_stopping_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward),
            "decision_making_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for shot stopping (based on goalkeeper's action to block shots)
            if o['right_team_roles'][o['active']] == 0 and o['game_mode'] in (6, 1):  # Penalty or Free Kick
                components["shot_stopping_reward"][rew_index] = 0.5
                reward[rew_index] += components["shot_stopping_reward"][rew_index]
            
            # Additional reward for effective communication (if the ball is in the penalty area and cleared)
            distance_from_goal = np.linalg.norm(o['ball'][:2] - np.array([1, 0]))
            if distance_from_goal < 0.2 and 'ball_owned_team' not in o:
                components["communication_reward"][rew_index] = 0.3
                reward[rew_index] += components["communication_reward"][rew_index]

            # Decision-making under pressure (quickly passing the ball away under pressure)
            if o['ball_owned_team'] == 0 and any(actions > 0 for actions in o['sticky_actions'][[0, 4]]):
                components["decision_making_reward"][rew_index] = 0.2
                reward[rew_index] += components["decision_making_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute one time step within the environment.
        This method should not change in the reward wrapper.
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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
