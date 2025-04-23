import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances learning of defensive and counterattack strategies in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveCheckpointRewards'] = self.defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle.get('DefensiveCheckpointRewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Reward defensive positioning and quick transition into counterattacks
        for idx, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                # Encouraging defensive actions
                if o['game_mode'] == 3:  # Free kick
                    reward[idx] += 0.1
                    components["defensive_positioning"][idx] += 0.1

                # Transition to counterattack
                if o['ball_owned_team'] == 0 and o['right_team_active'].any():
                    ball_x, ball_y = o['ball'][:2]
                    goal_x = 1.0  # Assuming the goal is on the right
                    distance_to_goal = np.sqrt((ball_x - goal_x)**2 + ball_y**2)
                    # Reward for moving ball towards opponent's goal
                    reward[idx] += max(0, 0.05 * (1 - distance_to_goal))
                    components["defensive_positioning"][idx] += max(0, 0.05 * (1 - distance_to_goal))
        
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
