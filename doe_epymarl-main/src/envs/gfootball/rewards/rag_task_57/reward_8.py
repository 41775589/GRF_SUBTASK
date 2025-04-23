import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on midfield-strategy coordination and offensive play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_positions = []
        self.striker_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_positions = []
        self.striker_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfielder_positions'] = self.midfielder_positions
        to_pickle['striker_positions'] = self.striker_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfielder_positions = from_pickle['midfielder_positions']
        self.striker_positions = from_pickle['striker_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        # Enhance reward function
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_role = o['active']
            player_pos = o['left_team'][player_role] if o['ball_owned_team'] == 0 else o['right_team'][player_role] 

            # Check if controlled player is a midfielder and moving to an offensive position
            if player_role in {4, 5, 6}:  # Midfield roles constants might require verification
                self.midfielder_positions.append(player_pos)
                if player_pos[0] > 0.5:  # Assuming positive x is offensive direction
                    reward[rew_index] += 0.1  # Reward for midfielders moving forward

            # Check if controlled player is a striker getting close to scoring
            if player_role in {9}:  # Striker role
                self.striker_positions.append(player_pos)
                distance_to_goal = abs(1 - player_pos[0])
                if distance_to_goal < 0.2:
                    reward[rew_index] += 0.5  # Higher reward the closer the striker is to the goal

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
