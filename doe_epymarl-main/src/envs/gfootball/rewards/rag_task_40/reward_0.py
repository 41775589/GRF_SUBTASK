import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive maneuvering and strategic positioning."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "strategic_position_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Assign a higher reward for effective defensive actions when the opponent has the ball.
            if o['ball_owned_team'] == 1:  # Assuming the agent's team is 0
                defensive_effectiveness = np.sum(o['left_team_active']) - np.sum(o.get('left_team_yellow_card', [0]*5))
                components["defensive_reward"][rew_index] = 0.05 * defensive_effectiveness

            # Check positioning for potential counterattacks
            # Reward positioning close to the ball when the ball is not owned by either team
            if o['ball_owned_team'] == -1:
                ball_position = np.array(o['ball'][0:2])
                player_position = np.array(o['left_team'][0])
                distance_to_ball = np.linalg.norm(player_position - ball_position)

                # Reward strategic positioning for counterattacks
                if distance_to_ball < 0.2:  # 20% of field proximity to the ball
                    components["strategic_position_reward"][rew_index] += 0.1

            reward[rew_index] += components["defensive_reward"][rew_index] + components["strategic_position_reward"][rew_index]

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_owner': self.previous_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_ball_owner = from_pickle['CheckpointRewardWrapper']['previous_ball_owner']
        return from_pickle

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
        return observation, reward, done, info
