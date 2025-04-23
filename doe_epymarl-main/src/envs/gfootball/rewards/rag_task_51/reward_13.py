import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on specialized goalkeeper training, including shot-stopping, quick reflexes, and initiating counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.saves_counter = 0
        self.passes_counter = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.saves_counter = 0
        self.passes_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['saves_counter'] = self.saves_counter
        state['passes_counter'] = self.passes_counter
        return state

    def set_state(self, state):
        self.saves_counter = state.get('saves_counter', 0)
        self.passes_counter = state.get('passes_counter', 0)
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "saving_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            goalie_index = np.argwhere(obs['left_team_roles'] == 0).flatten()[0]  # Index of the goalkeeper

            # Check if goalkeeper stops a goal
            if obs['ball_owned_team'] == 1 and obs['game_mode'] in [2, 3, 4, 6]:  # Modes where shots can occur
                # Assuming simple logic that any defence by goalkeeper is a save
                if obs['active'] == goalie_index and obs['ball_owned_player'] == goalie_index:
                    components["saving_reward"][rew_index] = 1.0
                    reward[rew_index] += components["saving_reward"][rew_index]
                    self.saves_counter += 1

            # Check if goalkeeper initiates counter-attacks
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == goalie_index:
                # Evaluate the effectiveness of the pass
                if obs['right_team_direction'][goalie_index][0] > 0:  # Assuming pass is towards the attacking side
                    components["passing_reward"][rew_index] = 0.5
                    reward[rew_index] += components["passing_reward"][rew_index]
                    self.passes_counter += 1

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
