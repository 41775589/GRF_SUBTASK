from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetBallPosition(-0.48, -0.06356)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.000000, 0.420000, Role.e_PlayerRole_GK)
  builder.AddPlayer(-0.422000, -0.19576, Role.e_PlayerRole_CB)
  builder.AddPlayer(-0.500000, 0.063559, Role.e_PlayerRole_CB, True)
  
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.000000, 0.420000, Role.e_PlayerRole_GK)
  builder.AddPlayer(-0.422000, -0.19576, Role.e_PlayerRole_CB)
  builder.AddPlayer(-0.500000, 0.063559, Role.e_PlayerRole_CB, True)
