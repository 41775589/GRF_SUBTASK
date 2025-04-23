import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specific for wingers focusing on 
    crossing and sprinting during soccer games."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        # Initialize reward components
        components["sprint_reward"] = [0.0] * len(reward)
        components["crossing_reward"] = [0.0] * len(reward)

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward strategy for sprints: Reward increasing sprints towards the cross-zones on the wings
            if 'right_team_roles' in o and o['right_team_roles'][o['active']] in (6, 7):  # Assuming role 6,7 are wingers
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # 'sticky_actions': 9 corresponds to sprint
                    # Normalize the reward based on how close to the crossing zone the player is.
                    y_pos = abs(o['right_team'][o['active']][1])  # y position of the controlled player
                    # Reward more the closer to the sidelines for crossing positions
                    components["sprint_reward"][rew_index] = (0.42 - y_pos) / 0.42  

            # Reward strategy for crosses: High reward for actions in the cross-zones
            if (o.get('game_mode', 0) == 4 or o['game_mode'] == 5):  # Assume modes 4 and 5 are crossing relevant modes
                components["crossing_reward"][rew_index] = 2.0

            # Update rewards
            reward[rew_index] += (components["sprint_reward"][rew_index] +
                                  components["crossing_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include reward components details in `info` for debugging purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
