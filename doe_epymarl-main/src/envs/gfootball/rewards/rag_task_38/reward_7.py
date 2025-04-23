import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents specializing in counterattacks through long passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_effectiveness_multiplier = 1.0
        self.transition_multiplier = 0.5
        self.total_counterattacking_rewards = {}
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_counterattacking_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.total_counterattacking_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.total_counterattacking_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "counterattack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_taken = o['sticky_actions']
            did_pass = action_taken[7]  # Assuming action index 7 is for long pass (action_number should be looked up)
            ball_owned_player = o['ball_owned_player']
            
            # Check if ball was passed from own half towards the counter-end
            ball_position_x = o['ball'][0]
            ball_moved_to_opponent_half = ball_position_x > 0 and o['ball_direction'][0] > 0
            
            # Reward only when ball is passed and moved to the opponent's half
            if did_pass and ball_owned_player == o['active'] and ball_moved_to_opponent_half:
                components['counterattack_reward'][rew_index] = self.pass_effectiveness_multiplier
                reward[rew_index] += components['counterattack_reward'][rew_index]
            
            # Check for quick transition by analyzing the reduction in active players back in defense
            if o['ball_owned_team'] == 0:  # assuming the team of concern is the left team here
                active_defender_positions = o['left_team'][o['left_team'][:, 0] < 0]  # defenders in own half
                if len(active_defender_positions) < 3:  # less than 3 defenders in the back suggests quick transition
                    reward[rew_index] += self.transition_multiplier
                    components['counterattack_reward'][rew_index] += self.transition_multiplier

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
