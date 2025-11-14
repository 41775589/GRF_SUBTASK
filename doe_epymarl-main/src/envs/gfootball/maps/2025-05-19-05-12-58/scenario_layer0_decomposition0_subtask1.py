from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(-0.7, 0.0)

    # Setting up the team on the left (defensive training for these agents)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, 0.1, e_PlayerRole_CB)  # Center Back 1 position for defensive practice
    builder.AddPlayer(-0.8, -0.1, e_PlayerRole_CB)  # Center Back 2 position for defensive practice

    # Setting up the opponent team on the right (attackers setup to challenge the defense)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.6, 0.12, e_PlayerRole_CF)  # Center Forward in an attacking position
    builder.AddPlayer(-0.6, -0.12, e_PlayerRole_RM)  # Right Midfielder providing cross
    builder.AddPlayer(-0.7, 0.0, e_PlayerRole_CM)  # Center Midfielder to assist in pressure
