import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward focused on shooting techniques."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Iterate over team players' observations
        for rew_index, o in enumerate(observation):
            if 'ball_owned_player' not in o or o['ball_owned_player'] != o['active']:
                continue
            
            # Criteria for shooting reward
            ball_pos = o['ball']
            if ball_pos[0] > 0.75:  # Ball is in the last quarter towards the opponent's goal
                distance_to_goal = abs(ball_pos[1])
                if distance_to_goal < 0.1 and o['game_mode'] == 0:  # close to the midline of goal and in normal game mode
                    components['shooting_reward'][rew_index] = 0.5

            reward[rew_index] += components['shooting_reward'][rew_index]

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

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)
