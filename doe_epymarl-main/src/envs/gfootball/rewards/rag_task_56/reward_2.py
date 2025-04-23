import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper specifically designed to improve the defensive capabilities
    of a team. It encourages goalkeepers to stop shots and initiate plays, and
    defenders to tackle efficiently and retain ball possession.

    This wrapper will give additional rewards for successful tackles,
    interceptions, and ball recoveries by defenders. Goalkeeper performance by
    stopping close shots and quickly passing the ball accurately will also be rewarded.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Custom reward parameters
        self.tackle_reward = 0.5
        self.interception_reward = 0.3
        self.ball_recovery_reward = 0.2
        self.goalkeeper_shot_stop_reward = 1.0
        self.goalkeeper_play_initiation_reward = 0.5

    def reset(self):
        """
        Reset the environment and the sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current state of the environment.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward function by adding extra rewards based on defensive actions.

        The reward is given for actions such as tackles, ball recoveries, and actions
        from the goalkeeper such as stopping a close shot or successfully initiating a play.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "extra_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Assumption: 'active' key indicates the active player and 'role' gives the player's role.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_role = o['left_team_roles' if o['ball_owned_team'] == 0 else 'right_team_roles'][o['active']]
            is_goalkeeper = player_role == 0   # Assuming role 0 is the goalkeeper

            if is_goalkeeper:
                if o['game_mode'] in [2, 3]:  # Close shot stopping modes (e.g., free kick, penalty)
                    components["extra_rewards"][rew_index] += self.goalkeeper_shot_stop_reward
                elif o['ball_owned_player'] == o['active']:
                    components["extra_rewards"][rew_index] += self.goalkeeper_play_initiation_reward
            elif player_role in [1, 2, 3, 4]:  # Assuming these roles are defenders
                if o['game_mode'] == 3:  # Tackle or ball recovery in a defensive mode
                    components["extra_rewards"][rew_index] += self.tackle_reward
                if o['ball_owned_player'] == o['active']:  # Defender retains control over the ball
                    components["extra_rewards"][rew_index] += self.ball_recovery_reward
            
            # Combine external and internal rewards
            reward[rew_index] += components["extra_rewards"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute step in the environment, augment reward information and return observations.

        Adds component values and final reward values to info for each step.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
