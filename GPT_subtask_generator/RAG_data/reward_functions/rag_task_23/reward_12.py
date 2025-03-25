import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for synergizing defensive roles between two agents in high-pressure scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        # Retrieve the observations to compute additional reward components
        observations = self.env.unwrapped.observation()
        if observations is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "defensive_synergy_reward": [0.0] * len(reward)}
        for idx, o in enumerate(observations):
            # Check if the teams are near their own penalty area under pressure
            player_pos = o['left_team'][o['active']]  # Active player position
            own_goal_x = -1
            opponent_goal_x = 1
            
            # Defensive synergy reward is triggered when close to own goal and under pressure
            if player_pos[0] < own_goal_x + 0.2:  # Close to own goal X-coord
                ball_owned_by_opponent = o.get('ball_owned_team', -1) == 1
                if ball_owned_by_opponent:
                    nearby_opponents = (np.linalg.norm(o['left_team'] - o['ball'], axis=1) < 0.1).sum()
                    if nearby_opponents >= 1:  # At least 1 opponent near the ball
                        components["defensive_synergy_reward"][idx] = 0.5  # Reward for defensive positioning

        # Modify reward based on the components
        for idx, base in enumerate(reward):
            reward[idx] += components["defensive_synergy_reward"][idx]

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
