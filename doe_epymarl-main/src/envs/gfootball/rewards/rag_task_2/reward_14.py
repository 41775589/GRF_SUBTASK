import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a teamwork reinforcement in defensive strategy using checkpoints."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        self._num_checkpoints = 8
        self._checkpoint_reward = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Track defensive coordination based on player positioning and ball possession
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Calculate the distance of each player to the ball
                distances = [np.linalg.norm(o['left_team'][i] - o['ball'][:2]) for i in range(len(o['left_team']))]
                
                # Find the closest player
                closest_player_id = np.argmin(distances)
                
                # Reward the closest player more and the second closest slightly less
                if closest_player_id == o['active']:
                    reward[rew_index] += self._checkpoint_reward 

                # Check for teamwork: are the players covering logical zones?
                distances_sorted_ids = np.argsort(distances)
                correct_positions = True
                for i in range(1, len(distances_sorted_ids)):
                    if distances[distances_sorted_ids[i]] < distances[distances_sorted_ids[i - 1]] + 0.1:
                        correct_positions = False
                        break

                if correct_positions:
                    components['checkpoint_reward'][rew_index] = self._checkpoint_reward
                    reward[rew_index] += components['checkpoint_reward'][rew_index] * 0.5

            # Penalty for losing the ball
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                reward[rew_index] -= 0.1

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
