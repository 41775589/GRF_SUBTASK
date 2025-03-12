from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.420000, Role.e_PlayerRole_GK, False)
  builder.AddPlayer(0.0, 0.020000, Role.e_PlayerRole_RM, False)
  builder.AddPlayer(0.2, -0.2, Role.e_PlayerRole_CF)
  
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.420000, Role.e_PlayerRole_GK, False)
  builder.AddPlayer(0.0, 0.020000, Role.e_PlayerRole_RM, False)
  builder.AddPlayer(0.2, -0.2, Role.e_PlayerRole_CF)
