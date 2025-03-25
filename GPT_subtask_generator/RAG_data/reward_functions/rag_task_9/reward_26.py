import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on offensive skills such as passing, shooting, and dribbling."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
                      "offensive_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Evaluating offensive skills from observations
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            player_pos = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            own_goal_pos = [1, 0] if o['ball_owned_team'] == 1 else [-1, 0]

            # If the active player has the ball, compute skills reward
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] != -1:
                distance_to_goal = np.linalg.norm(np.array(ball_pos[:2]) - np.array(own_goal_pos))
                control_bonus = 0.1  # Default control bonus for having the ball
                components['offensive_skill_reward'][rew_index] += control_bonus

                # Reward for approaching opponent's goal with the ball
                if distance_to_goal < 0.5:  # threshold for distance to goal to consider close
                    components['offensive_skill_reward'][rew_index] += 0.5

                # Encourage passing closer to opponent goal or taking a shot
                if 'action' in o:
                    if o['action'][8] == 1:  # is_sprint
                        components['offensive_skill_reward'][rew_index] += 0.05
                    if o['action'][9] == 1:  # is_dribble
                        components['offensive_skill_reward'][rew_index] += 0.05
                    if o['action'][7] == 1:  # shot
                        components['offensive_skill_reward'][rew_index] += 0.1
                    if o['action'][6] == 1:  # long_pass
                        components['offensive_skill_reward'][rew_index] += 0.1
                    if o['action'][5] == 1:  # short_pass
                        components['offensive_skill_reward'][rew_index] += 0.1

            # Update the base reward with the offensive skills reward components
            reward[rew_index] += components['offensive_skill_reward'][rew_index]

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
