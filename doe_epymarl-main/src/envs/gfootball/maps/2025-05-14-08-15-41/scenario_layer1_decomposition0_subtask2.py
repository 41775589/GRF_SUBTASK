from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.75, 0.0, e_PlayerRole_CB)  # Main defensive player
  builder.AddPlayer(-0.75, 0.1, e_PlayerRole_DM)  # Supports defense and starts repositioning

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.7, 0.0, e_PlayerRole_CF)   # Opposing center forward to challenge defense
  builder.AddPlayer(-0.7, 0.05, e_PlayerRole_CM)  # Opposing midfielder to create dynamic play
