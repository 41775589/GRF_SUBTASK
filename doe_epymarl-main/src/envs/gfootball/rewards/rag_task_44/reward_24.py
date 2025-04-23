import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for effectively managing the Stop-Dribble under pressure.
    The reward function encourages maintaining ball possession and executing Stop-Dribble action when the opposing players are close.
    """
    def __init__(self, env):
        super().__init__(env)
        # The threshold distance to consider a player is under pressure
        self._pressure_distance = 0.1
        # Reward amount for successfully stopping the dribble under pressure
        self._stop_dribble_reward = 1.0
        self._num_actions = 10
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy()}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        components["stop_dribble_reward"] = [0.0] * len(reward)

        for i, obs in enumerate(observation):
            # Detection of action Stop-Dribble and teams in dribbling state
            dribbling = obs['sticky_actions'][9] == 1
            stop_dribble_action = obs['sticky_actions'][8] == 1
            player_pos = obs['right_team' if obs['designated'] > 0 else 'left_team'][obs['active']]
            
            if dribbling and stop_dribble_action:
                # Check minimum distance to any opponent to determine pressure
                opponent_team = 'left_team' if obs['designated'] > 0 else 'right_team'
                distances = np.linalg.norm(obs[opponent_team] - player_pos, axis=1)
                under_pressure = np.any(distances < self._pressure_distance)

                if under_pressure:
                    components["stop_dribble_reward"][i] = self._stop_dribble_reward
                    reward[i] += self._stop_dribble_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        # Update actions counter from the current state of the environment
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state

        return observation, reward, done, info
