import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages offensive strategies, including accurate shooting, effective dribbling,
    and mastering different pass types through additional rewards contingent on observable criteria.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.accuracy_bonus = 0.2  # Reward bonus for close shooting.
        self.dribbling_bonus = 0.3 # Reward bonus for successful dribbling.
        self.passing_bonus = 0.1   # Reward bonus for long and high passes.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['extra_state_info'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['extra_state_info']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "accuracy_bonus": [0.0] * len(reward),
                      "dribbling_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, (rew, obs) in enumerate(zip(reward, observation)):
            if obs['ball_owned_team'] == 1:  # We assume 1 for the controlled/observed team
                # Close shots reward: ball near the opponent's goal posts with a shooting action
                if np.abs(obs['ball'][0] - 1) < 0.05 and obs['sticky_actions'][6]:  # Action 'shoot'
                    components["accuracy_bonus"][i] = self.accuracy_bonus
                    rew += components["accuracy_bonus"][i]

                # Successful dribbling reward: ball with owner and quick change in direction
                if obs['sticky_actions'][9]:  # Action 'dribble'
                    velocity = np.linalg.norm(obs['ball_direction'][:2])
                    if velocity > 0.01:  # arbitrary velocity threshold indicating successful dribble
                        components["dribbling_bonus"][i] = self.dribbling_bonus
                        rew += components["dribbling_bonus"][i]

                # Reward for high and long passes: if pass actions are performed over distance
                if obs['sticky_actions'][1] or obs['sticky_actions'][2]:  # Action 'high' or 'long' pass
                    components["passing_bonus"][i] = self.passing_bonus
                    rew += components["passing_bonus"][i]

            reward[i] = rew

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
            for j, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{j}"] = action
        return observation, reward, done, info
