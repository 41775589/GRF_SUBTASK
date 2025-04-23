import gym
import numpy as np
class GoalkeeperTrainingRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically for goalkeeper training."""

    def __init__(self, env):
        super(GoalkeeperTrainingRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define the parameters for the reward function
        self.save_reward = 1.0  # Reward for stopping a shot
        self.pass_reward = 0.5  # Reward for initiating a counter-attack with a pass
        self.positioning_reward = 0.1  # Incremental reward for positioning closer to the goal line

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['GoalkeeperTrainingRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Access the unwrapped environment's current observation
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), 
                      "save_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            goalie_position = o['left_team'][0]  # Assuming goalie is the first player in left_team
            goalie_role = o['left_team_roles'][0]

            # Check if the player is the goalkeeper and positioned correctly
            if goalie_role == 0:
                # Improve positioning reward based on how close to the goal line the goalie is
                # Goal line for the left team goalkeeper is at x = -1
                distance_to_goal_line = abs(goalie_position[0] + 1)
                components["positioning_reward"][rew_index] = self.positioning_reward * (1 - distance_to_goal_line)

            # Check if a shot was stopped
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0:
                # Assume game mode 6 (Penalty) to simulate shot-stopping in training
                if o['game_mode'] == 6:
                    components["save_reward"][rew_index] = self.save_reward

            # Reward counter-attack initiating pass
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if o['ball_owned_player'] == 0 and o['sticky_actions'][9] == 1:  # action_dribble initiates a pass
                    components["pass_reward"][rew_index] = self.pass_reward

            # Update the cumulative reward for this player
            reward[rew_index] += (components["save_reward"][rew_index] + 
                                  components["pass_reward"][rew_index] + 
                                  components["positioning_reward"][rew_index])

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
