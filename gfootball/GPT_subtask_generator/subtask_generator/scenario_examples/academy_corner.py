"""Standard corner-kick situation, except that the corner taker can
run with the ball from the corner. The episode does not end if
possession is lost."""

from . import *

def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False

  builder.SetBallPosition(0.99, 0.41)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(1.0, 0.42, e_PlayerRole_LB)
  builder.AddPlayer(0.7, 0.15, e_PlayerRole_CB)
  builder.AddPlayer(0.7, 0.05, e_PlayerRole_CB)
  builder.AddPlayer(0.7, -0.05, e_PlayerRole_RB)
  builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM)
  builder.AddPlayer(0.6, 0.35, e_PlayerRole_CM)
  builder.AddPlayer(0.8, 0.07, e_PlayerRole_CM)
  builder.AddPlayer(0.8, -0.03, e_PlayerRole_LM)
  builder.AddPlayer(0.8, -0.13, e_PlayerRole_RM)
  builder.AddPlayer(0.7, -0.3, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.75, -0.18, e_PlayerRole_LB)
  builder.AddPlayer(-0.75, -0.08, e_PlayerRole_CB)
  builder.AddPlayer(-0.75, 0.02, e_PlayerRole_CB)
  builder.AddPlayer(-1.0, -0.1, e_PlayerRole_RB)
  builder.AddPlayer(-0.8, -0.25, e_PlayerRole_CM)
  builder.AddPlayer(-0.88, -0.07, e_PlayerRole_CM)
  builder.AddPlayer(-0.88, 0.03, e_PlayerRole_CM)
  builder.AddPlayer(-0.88, 0.13, e_PlayerRole_LM)
  builder.AddPlayer(-0.75, 0.25, e_PlayerRole_RM)
  builder.AddPlayer(-0.2, 0.0, e_PlayerRole_CF)
