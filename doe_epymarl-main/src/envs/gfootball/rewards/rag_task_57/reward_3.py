import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes offensive plays between midfielders and strikers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_ownership_reward = 0.2
        self.striker_finish_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.sticky_actions_counter = from_picle['sticky_actions']
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_ownership_reward": 0.0,
                      "striker_finish_reward": 0.0}

        if observation is None:
            return reward, components

        for agent_id, agent_obs in enumerate(observation):
            active_player_idx = agent_obs.get('active')
            if active_player_idx is None:
                continue

            # Check if a midfielder has completed a pass
            if agent_obs['left_team_roles'][active_player_idx] in [6, 7, 8]:  # Midfield roles
                if agent_obs['ball_owned_player'] == active_player_idx:
                    components["midfield_ownership_reward"] = self.midfield_ownership_reward
                    reward[agent_id] += components["midfield_ownership_reward"]
            
            # Check if a striker has finished
            if agent_obs['left_team_roles'][active_player_idx] == 9:  # Striker role
                if agent_obs['score'][0] > 0:  # Assuming score for left team
                    components["striker_finish_reward"] = self.striker_finish_reward
                    reward[agent_id] += components["striker_finish_reward"]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
