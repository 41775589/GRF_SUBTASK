import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to incentivize collaborative plays between shooters and passers to exploit scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_holder = None
        self.collaborative_play_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_holder = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        to_pickle['previous_ball_holder'] = self.previous_ball_holder
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        self.previous_ball_holder = from_pickle.get('previous_ball_holder', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaborative_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            team_has_ball = o['ball_owned_team'] == o['designated']//11
            ball_holder = o['ball_owned_player']
            if team_has_ball and ball_holder != -1:
                is_goal_scored = reward[rew_index] > 0  # Check if a goal was scored
                if is_goal_scored and self.previous_ball_holder is not None and self.previous_ball_holder != ball_holder:
                    # Collaborative play detected: previous ball holder passed the ball leading to a goal
                    components["collaborative_play_reward"][rew_index] = self.collaborative_play_reward
                    reward[rew_index] += components["collaborative_play_reward"][rew_index]

            self.previous_ball_holder = ball_holder if team_has_ball else None
        
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
