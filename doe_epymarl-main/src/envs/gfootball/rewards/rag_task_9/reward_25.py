import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for encouraging offensive skills such as passing,
    shooting, dribbling, sprinting, and creating scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define weights for each offensive action to tune their incentive level
        self.pass_weight = 0.1
        self.shot_weight = 0.2
        self.dribble_weight = 0.1
        self.sprint_weight = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_skills_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = components["base_score_reward"][rew_index]
            offensive_reward = 0.0

            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # Team 0 is left, and active player has the ball
                # Check proximity to the goal for shot reward
                goal_distance = 1 - o['ball'][0]  # X position of the ball, assuming goal at x = 1
                if goal_distance < 0.3:  # Close to the goal
                    offensive_reward += self.shot_weight
                
                # Check if doing a dribble or sprint
                if o['sticky_actions'][8] == 1:  # Sprint
                    offensive_reward += self.sprint_weight
                if o['sticky_actions'][9] == 1:  # Dribble
                    offensive_reward += self.dribble_weight

                # Assuming effective passing strategies, position change triggers pass rewards
                if np.linalg.norm(o['ball_direction'][:2]) > 0.02:  # Significant ball movement
                    if abs(o['ball_direction'][0]) > abs(o['ball_direction'][1]):  # Horizontal direction predominates
                        offensive_reward += self.pass_weight  # This can be both short or long pass

            reward[rew_index] = base_reward + offensive_reward
            components["offensive_skills_reward"][rew_index] = offensive_reward

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
