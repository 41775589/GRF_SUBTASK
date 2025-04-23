import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for effective defensive maneuvers and positioning.
    The auxiliary reward contributes for intercepting direct attacks, blocking opponent trajectories,
    and positioning effectively for potential counterattacks.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for defensive reward
        self.interception_reward = 0.3
        self.block_reward = 0.2
        self.positioning_reward = 0.1
        self._reset_defensive_stats()

    def _reset_defensive_stats(self):
        # Trackers for defensive statistics
        self.interceptions = np.zeros(2, dtype=int)
        self.blocks = np.zeros(2, dtype=int)
        self.positioned_correctly = np.zeros(2, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._reset_defensive_stats()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['interceptions'] = self.interceptions
        to_pickle['blocks'] = self.blocks
        to_pickle['positioned_correctly'] = self.positioned_correctly
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle.get('interceptions', np.zeros(2, dtype=int))
        self.blocks = from_pickle.get('blocks', np.zeros(2, dtype=int))
        self.positioned_correctly = from_pickle.get('positioned_correctly', np.zeros(2, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": [0.0]*2, 
                      "block_reward": [0.0]*2, 
                      "positioning_reward": [0.0]*2}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for interceptions
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] != o['active']:
                components['interception_reward'][rew_index] = self.interception_reward
                self.interceptions[rew_index] += 1
            
            # Simulated defensive block reward
            opponent = (rew_index + 1) % 2  # Assuming only two agents for simplicity
            opponent_dist_to_goal = np.linalg.norm(o['right_team'][opponent] - np.array([1, 0]))
            if opponent_dist_to_goal < 0.1 and not o['right_team_active'][opponent]:  # If opponent is close to scoring but inactive
                components['block_reward'][rew_index] = self.block_reward
                self.blocks[rew_index] += 1

            # If positioned in a defensively strategic place
            if np.linalg.norm(o['left_team'][o['active']] - np.array([-1, 0])) < 0.3:  # Close to own goal
                components['positioning_reward'][rew_index] = self.positioning_reward
                self.positioned_correctly[rew_index] += 1

            reward[rew_index] += sum(components[key][rew_index] for key in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
