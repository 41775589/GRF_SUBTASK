import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for defensive counter-play training."""

    def __init__(self, env):
        super().__init__(env)
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
        """
        Augments the default reward system with a new reward scheme focusing on defensive actions and coordination.
        Defensive actions, player positioning relative to an attacking player, and successful ball dispossession
        yield additional rewards.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reinforce agents to stay closer to the goal area when defending
            if o['ball_owned_team'] == 1 and o['left_team_active'][o['active']]:
                # Position closer to own goal adds rewards
                x_pos = o['left_team'][o['active']][0]
                if x_pos < 0:
                    components["defensive_reward"][rew_index] = 0.3 * (1 + abs(x_pos))
                    reward[rew_index] += components["defensive_reward"][rew_index]

            # Encourage dispossessing the ball
            if o['ball_owned_team'] == 1 and o['left_team_roles'][o['active']] == 1:  # 1 represents defenders in roles
                components["defensive_reward"][rew_index] += 0.5
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Promote good passing under pressure
            if o['left_team_active'][o['active']] and \
               ('action_high_pass' in o['sticky_actions'] or 'action_long_pass' in o['sticky_actions']):
                components["defensive_reward"][rew_index] += 0.2
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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
