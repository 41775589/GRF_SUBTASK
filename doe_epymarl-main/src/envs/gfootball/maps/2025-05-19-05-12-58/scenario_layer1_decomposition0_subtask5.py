from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    # Set ball starting position near the midfield area to facilitate midfield play
    builder.SetBallPosition(-0.5, 0.0)

    # Set up the left team (training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CM, controllable=True)  # Controlled midfielder to practice passes
    
    # Set up a minimal opposition to simulate defensive transitions
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.7, 0.15, e_PlayerRole_LM)
    builder.AddPlayer(-0.7, -0.15, e_PlayerRole_RM)
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_CB)
