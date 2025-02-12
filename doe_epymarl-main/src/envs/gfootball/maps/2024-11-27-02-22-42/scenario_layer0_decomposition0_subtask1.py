from . import *
def build_scenario(builder):
    builder.config().game_duration = 800  # Longer duration for more complex defensive scenarios
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Keep the game going even after possession changes

    builder.SetBallPosition(0.0, 0.0)  # Start with the ball in the middle

    # Defining the defensive team (Left Team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CB)  # Centre Back 1
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_CB)  # Centre Back 2

    # Adding the two main players focused on defensive skills
    builder.AddPlayer(-0.4, 0.15, e_PlayerRole_DM)  # Defensive Midfielder 1 - Main training agent
    builder.AddPlayer(-0.4, -0.15, e_PlayerRole_DM)  # Defensive Midfielder 2 - Main training agent

    # Opposing team (Right Team) setup to simulate attacking scenarios
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    builder.AddPlayer(0.4, 0.2, e_PlayerRole_CF)  # Forward player 1
    builder.AddPlayer(0.4, -0.2, e_PlayerRole_CF)  # Forward player 2
    builder.AddPlayer(0.3, 0.0, e_PlayerRole_AM)  # Attacking Midfielder

    # Set up to induce a lot of defensive interactions
    builder.AddPlayer(0.2, 0.1, e_PlayerRole_CM)  # Additional Midfielder 1
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_CM)  # Additional Midfielder 2
