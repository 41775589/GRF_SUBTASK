from . import *
def build_scenario(builder):
    # General configuration for the scenario duration and randomness
    builder.config().game_duration = 3000
    builder.config().deterministic = False
    builder.config().right_team_difficulty = 0.05
    builder.config().left_team_difficulty = 0.05
   
    # Setting which team starts with the ball based on episode number
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left

    builder.SetTeam(first_team)

    # Adding two defensive players with a focus on different skills
    # First player as Left Back - focus on sliding and positional play
    builder.AddPlayer(-0.7, 0.15, e_PlayerRole_LB)
    # Second player as Center Back - focus on stamina management and intercepting passes
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)

    # Setting the goalkeeper for the team, not a main player for the current training but necessary for team setup
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)

    builder.SetTeam(second_team)
    
    # Adding opponent players for realistic defensive scenarios
    # More players in attack simulate defensive scenarios better
    builder.AddPlayer(-0.5, 0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.5, -0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_AM)

    # Setting the goalkeeper for the right team, also not controllable
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
