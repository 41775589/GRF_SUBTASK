import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for goalkeeper coordination and efficient ball clearing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_coordination_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Reward Components for Goalkeeper Coordination
        for i in range(len(reward)):
            player_obs = observation[i]
            if player_obs['active'] == player_obs['left_team_roles'][0]:  # Assuming goalkeeper is at index 0
                if player_obs['game_mode'] in (3, 4):  # Free kick or corner
                    components['goalkeeper_coordination_reward'][i] = 0.2  # Increase reward for defending these modes
                # Clearing the ball to specific outfield players efficiently
                if player_obs['ball_owned_team'] == 0:  # Assuming team 0 is the left team (defensive)
                    components['goalkeeper_coordination_reward'][i] += 0.1  # Reward for having the ball
                    if player_obs['ball_direction'][0] > 0:  # Assuming that positive x is towards the opponent's goal
                        components['goalkeeper_coordination_reward'][i] += 0.1  # Clearing the ball towards players efficiently
            reward[i] += sum(components.values())

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
            for i, action_item in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_item
        return observation, reward, done, info
