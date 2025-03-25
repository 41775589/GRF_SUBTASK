import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward focusing on dribbling and sprinting skills 
    in tight spaces against defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Assuming equal number of regions and checkpoints equaling number of players not including the goalie.
        self.num_dribble_points = 4  # Points for dribble checkpoints
        self.dribble_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle=None):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = state['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_bonus": [0.0] * len(reward),
            "sprint_bonus": [0.0] * len(reward)
        }

        for index, (rew, obs) in enumerate(zip(reward, observation)):
            # Tracking dribbling quality through control zones
            if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:  # Left team
                ball_x = abs(obs['ball'][0])  # Simple simplification for demo
                if ball_x > 0.8:
                    components['dribbling_bonus'][index] = self.dribble_reward
                reward[index] += components['dribbling_bonus'][index]

            # Reward engaging sprint action in offensive plays
            if obs['sticky_actions'][8] == 1:  # Sprint action index
                components['sprint_bonus'][index] = self.num_dribble_points * self.dribble_reward / 2  # Half the dribble reward
                reward[index] += components['sprint_bonus'][index]

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
