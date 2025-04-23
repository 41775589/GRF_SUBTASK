import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that specializes in strategic support for goalkeeper, 
    penalizing loss of control by the goalie under pressure, and rewarding 
    effective ball clearances to specific outfield players.
    """
    def __init__(self, env):
        super().__init__(env)
        self._goalie_backup = 0.5   # Reward for goalie clearing the ball under high pressure
        self._penalty_lost_control = -1  # Penalty if goalie loses control under pressure
        self._reward_pass_to_mate = 0.3  # Reward for passing the ball to specific mate
        self.ball_last_owned_by_goalie = False
        self.goalkeeper_index = None  # Will determine dynamically
        self.designated_targets = set()  # Set of teammate indices to reward passes to
        self.prepare_designated_players()
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To track sticky actions

    def reset(self):
        """
        Reset for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = None
        self.prepare_designated_players()
        self.ball_last_owned_by_goalie = False
        return self.env.reset()

    def reward(self, reward):
        """
        Adjust the reward based on goalie's gameplay.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalie_backup_reward": 0.0,
                      "penalty_lost_control": 0.0,
                      "reward_pass_to_mate": 0.0}

        if observation is None:
            return reward, components

        # Determine the goalkeeper index dynamically if not set
        if self.goalkeeper_index is None:
            for idx, role in enumerate(observation[0]['left_team_roles']):
                if role == 0:  # e_PlayerRole_GK
                    self.goalkeeper_index = idx
                    break

        for team_side in ['left_team', 'right_team']:
            if observation[0]['ball_owned_team'] == 0 and team_side == 'left_team':  # ball owned by left team
                if observation[0]['ball_owned_player'] == self.goalkeeper_index:
                    self.ball_last_owned_by_goalie = True
                elif self.ball_last_owned_by_goalie:
                    # If ball was last owned by goalie and now owned by another player
                    if observation[0]['ball_owned_player'] in self.designated_targets:
                        components['reward_pass_to_mate'] += self._reward_pass_to_mate
                    self.ball_last_owned_by_goalie = False
            elif self.ball_last_owned_by_goalie and observation[0]['ball_owned_team'] != 0:
                # If goalie loses control to the other team
                components['penalty_lost_control'] += self._penalty_lost_control
                self.ball_last_owned_by_goalie = False

        # Update the simple reward with our adjusted components
        final_reward = reward + components['goalie_backup_reward'] \
                             + components['penalty_lost_control'] \
                             + components['reward_pass_to_mate']

        return final_reward, components

    def prepare_designated_players(self):
        """
        Prepare the designated players index which will be dynamically rewarded for 
        receiving passes from the goalie. These would typically be midfielders or defenders.
        """
        # Placeholder: might want a better strategy to designate targets
        self.designated_targets = {1, 2, 3, 4}  # represents player indices

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
