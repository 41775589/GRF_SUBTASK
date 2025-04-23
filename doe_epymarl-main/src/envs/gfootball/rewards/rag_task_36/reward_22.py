import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages dribbling skills and dynamic transitions between defense and offense."""

    def __init__(self, env, dribble_reward=0.2, transition_reward=0.3):
        gym.RewardWrapper.__init__(self, env)
        self.dribble_reward = dribble_reward
        self.transition_reward = transition_reward
        self.last_ball_position = None
        self.last_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_position = None
        self.last_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position,
            'last_ball_owned_team': self.last_ball_owned_team
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        self.last_ball_owned_team = from_pickle['CheckpointRewardWrapper']['last_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_position = o['ball']
            current_ball_owned_team = o['ball_owned_team']
            
            if self.last_ball_position is not None:
                # Calculate the distance the ball moved since the last action
                dist_moved = np.linalg.norm(np.array(current_ball_position[:-1]) - np.array(self.last_ball_position[:-1]))
                # Reward for dribbling: if the ball is owned by the same team and is moving
                if current_ball_owned_team == self.last_ball_owned_team and dist_moved > 0.05:
                    components["dribble_reward"][rew_index] = self.dribble_reward
            
            # Reward for successful transition: if possession changes to us near our goal area or if we just scored
            if self.last_ball_owned_team is not None and current_ball_owned_team != self.last_ball_owned_team:
                if current_ball_owned_team == 0 and current_ball_position[0] < -0.5:
                    components["transition_reward"][rew_index] = self.transition_reward
                elif reward[rew_index] > 0:  # our team scored
                    components["transition_reward"][rew_index] = self.transition_reward

            reward[rew_index] += components["dribble_reward"][rew_index] + components["transition_reward"][rew_index]
            # Update last known positions
            self.last_ball_position = current_ball_position
            self.last_ball_owned_team = current_ball_owned_team

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
