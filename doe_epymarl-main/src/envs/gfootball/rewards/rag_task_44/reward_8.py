import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_stickies = [None, None]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_team = None
        self.previous_stickies = [None, None]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_owned_team'] = self.previous_ball_owned_team
        to_pickle['previous_stickies'] = self.previous_stickies
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owned_team = from_pickle.get('previous_ball_owned_team', None)
        self.previous_stickies = from_pickle.get('previous_stickies', [None, None])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0, 0.0]
        }

        for i in range(len(reward)):
            o = observation[i]
            components["stop_dribble_reward"][i] = 0

            # Check if team ownership of the ball changed
            if self.previous_ball_owned_team is not None and o['ball_owned_team'] != self.previous_ball_owned_team:
                # If the current team just took possession, potentially reward based on previous sticky actions
                if self.previous_stickies[i] is not None:
                    # Check the sticky action flags for the stop dribble pattern
                    if self.previous_stickies[i][9] and not o['sticky_actions'][9]:  # Dribble was stoped
                        components["stop_dribble_reward"][i] = 1.0
                # Reset the previous stickies when the ball possession changes
                self.previous_stickies[i] = None
            else:
                # Update the records if the ball is still owned by the same team.
                self.previous_stickies[i] = np.copy(o['sticky_actions'])

            self.previous_ball_owned_team = o['ball_owned_team']
            reward[i] += components["stop_dribble_reward"][i]
        
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
