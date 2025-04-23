import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper for optimizing high pass strategies from the midfield,
    particularly focusing on perfecting timing and placement for creating direct scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_rewards = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track uses of sticky actions.
        # High pass has a high reward if it results in a goal-scoring opportunity.
        self.high_pass_reward = 0.2  # Incentive for making effective high passes.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Make base score reward and pass reward.
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Check both agents.
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Only reward high passes from midfielders (roles 4,5,6) aiming at forward players.
            if o['active'] in [4, 5, 6] and o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                target_player_role = o['right_team_roles'][o['designated']]

                # Check if the ball was passed high.
                if o['sticky_actions'][6] == 1 and np.abs(o['ball_direction'][2]) > 0.3:  # High pass action is 6.
                    target_pos = o['right_team'][o['designated']]

                    # Check if the pass is towards an attacking role.
                    if target_player_role in [8, 9]:  # AM and CF roles.
                        distance_to_goal = 1 - target_pos[0]
                        
                        # Reward based on how direct the play is to the goal (1 being right in front of the goal).
                        if distance_to_goal < 0.5:
                            components["pass_reward"][rew_index] = self.high_pass_reward
                            reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions usage.
        observation = self.env.unwrapped.observation()
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return obs, reward, done, info
