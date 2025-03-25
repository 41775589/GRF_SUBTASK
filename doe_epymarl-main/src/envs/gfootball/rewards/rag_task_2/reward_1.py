import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive coordination and teamwork dense reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.positional_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positional_rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            current_reward = reward[rew_index]
            if o['ball_owned_team'] == 0:
                ball_player_pos = np.array(o['left_team'][o['ball_owned_player']])
                own_goal_pos = np.array([-1, 0])
                dist_to_goal_own = np.linalg.norm(ball_player_pos - own_goal_pos)

                if dist_to_goal_own < 0.5:
                    # Defensive position close to own goal
                    reward_key = (rew_index, 'close_defensive')
                    if reward_key not in self.positional_rewards_collected:
                        components['defensive_coordination_reward'][rew_index] += 0.02
                        self.positional_rewards_collected[reward_key] = True
            else:
                # Encouraging players to move towards the ball when not in possession
                ball_pos = np.array(o['ball'][:2])
                players_pos = np.array(o['left_team'])
                dists_to_ball = np.linalg.norm(players_pos - ball_pos, axis=1)
                min_dist_to_ball = np.min(dists_to_ball)

                if min_dist_to_ball < 0.3:
                    reward_key = (rew_index, 'ball_chasing')
                    if reward_key not in self.positional_rewards_collected:
                        components['defensive_coordination_reward'][rew_index] += 0.05
                        self.positional_rewards_collected[reward_key] = True
           
            reward[rew_index] += components['defensive_coordination_reward'][rew_index]

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
