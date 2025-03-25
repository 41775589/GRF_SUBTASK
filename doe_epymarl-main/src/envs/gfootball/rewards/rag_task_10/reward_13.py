import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on defensive actions.
    Reward is based on effective defensive positioning, interceptions, and tackling.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Defend actions include stopping opponents' dribble or movement.
            active_actions = set(o['sticky_actions'][8:10])  # Indices for sprint and dribble actions
            opponent_close = False
            # Check if opponents are close to the controlled player
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                player_pos = o['right_team'][o['active']]
                for opponent_pos in o['left_team']:
                    if np.linalg.norm(player_pos - opponent_pos) < 0.05:
                        opponent_close = True
                        break

            # Reward interception or block attempts when opponents are close
            if opponent_close and any(active_actions):
                components["defensive_reward"][rew_index] = 0.2  # Reward for good defensive positioning/action
                
            # Update the main reward with the defensive reward component
            reward[rew_index] += components["defensive_reward"][rew_index]

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
