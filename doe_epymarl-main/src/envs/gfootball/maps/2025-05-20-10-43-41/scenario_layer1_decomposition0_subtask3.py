from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
  
    # Set initial Ball Position for a scenario focusing on counter-attacks
    builder.SetBallPosition(0.5, 0.0)

    # Left team (our team trained for offensive counter-attack skills)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_RM, controllable=True)  # Controllable attacking midfielder initiating counter-attacks
  
    # Right team (defensive players)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.25, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.25, 0.1, e_PlayerRole_CB)
