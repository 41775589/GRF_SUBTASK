import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing mid to long-range passing effectiveness and adds rewards for precision and usage in strategic coordination plays."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_distance_threshold = 0.5  # threshold to consider a pass as long range
        self.pass_precision_reward = 0.2
        self.coordinated_play_bonus = 0.3

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
                      "pass_precision_reward": [0.0] * len(reward),
                      "coordinated_play_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if a long-range pass has been made
            if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > self.pass_distance_threshold:
                if 'ball_owned_team' in o and o['ball_owned_team'] != -1:
                    # Increase rewards for precision passes that change possession within the team
                    if 'ball_owned_player' in o and o['designated'] != o['ball_owned_player']:
                        reward[rew_index] += self.pass_precision_reward * 1.5  # more reward if ball is received by another team player

                # If the long pass results in a strategic positioning closer to the goal
                previous_distance_to_goal = np.abs(o['ball'][0] - 1)  # Assume working with the opponent's goal at x = 1
                current_distance_to_goal = np.abs(o['ball'][0] + o['ball_direction'][0] - 1)
                if current_distance_to_goal < previous_distance_to_goal:
                    components["coordinated_play_bonus"][rew_index] = self.coordinated_play_bonus
                    reward[rew_index] += self.coordinated_play_bonus

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
