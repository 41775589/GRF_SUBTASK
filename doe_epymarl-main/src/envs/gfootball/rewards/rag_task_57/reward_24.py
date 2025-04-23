import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on midfielders' coordination in space creation and ball delivery,
    along with strikers' finishing plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components
        
        # Analyze each agent's state independently
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for successful passes between midfielders and forwards near the opponent's goal area
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0:
                if o['right_team_roles'][o['active']] in [8, 9]:  # Verify if the agent is a midfielder or striker
                    distance_to_goal = 1 - abs(o['ball'][0])  # Goal is at x = ±1
                    if distance_to_goal < 0.2:
                        # High reward for productive actions close to the goal
                        bonus = 0.3 * (0.2 - distance_to_goal)
                        reward[rew_index] += bonus
                        components.setdefault('goal_zone_bonus', []).append(bonus)
                    else:
                        components.setdefault('goal_zone_bonus', []).append(0.0)
                else:
                    components.setdefault('goal_zone_bonus', []).append(0.0)
            else:
                components.setdefault('goal_zone_bonus', []).append(0.0)

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
        return observation, reward, done, info
