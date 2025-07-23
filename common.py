import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px


pd.set_option('display.max_columns', None)
plt.style.use('dark_background')
pio.templates.default = "plotly_dark"
pio.renderers.default = "jupyterlab+png"


TEAM_ABBREVIATIONS = [
    'CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WSH',
    'BOS', 'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TBL', 'TOR',
    'CHI', 'COL', 'DAL', 'MIN', 'NSH', 'STL', 'UTA', 'WPG',
    'ANA', 'CGY', 'EDM', 'LAK', 'SEA', 'SJS', 'VAN', 'VGK',
]

TEAM_NAMES = [
    # Metropolitan Division
    "carolina-hurricanes", "columbus-blue-jackets", "new-jersey-devils", "new-york-islanders",
    "new-york-rangers", "philadelphia-flyers", "pittsburgh-penguins", "washington-capitals",

    # Atlantic Division
    "boston-bruins", "buffalo-sabres", "detroit-red-wings", "florida-panthers",
    "montreal-canadiens", "ottawa-senators", "tampa-bay-lightning", "toronto-maple-leafs",

    # Central Division
    "chicago-blackhawks", "colorado-avalanche", "dallas-stars", "minnesota-wild",
    "nashville-predators", "st.-louis-blues", "utah-mammoth", "winnipeg-jets",

    # Pacific Division
    "anaheim-ducks", "calgary-flames", "edmonton-oilers", "los-angeles-kings",
    "san-jose-sharks", "seattle-kraken", "vancouver-canucks", "vegas-golden-knights"
]

TEAM_COLORS = {
    "Anaheim Ducks": "#F47A38",
    "Arizona Coyotes": "#751824",
    "Atlanta Thrashers": "#77C8ED",
    "Boston Bruins": "#FFB81C",
    "Buffalo Sabres": "#F1C80D",
    "Calgary Flames": "#E8210B",
    "Carolina Hurricanes": "#CC0000",
    "Chicago Blackhawks": "#D30404",
    "Colorado Avalanche": "#5A172C",
    "Columbus Blue Jackets": "#002654",
    "Dallas Stars": "#16A32B",
    "Detroit Red Wings": "#CE1126",
    "Edmonton Oilers": "#BB4817",
    "Florida Panthers": "#DF1414",
    "Los Angeles Kings": "#969696",
    "Minnesota Wild": "#13573D",
    "Montréal Canadiens": "#AF1E2D",
    "Nashville Predators": "#FFB81C",
    "New Jersey Devils": "#CE1126",
    "New York Islanders": "#F29514",
    "New York Rangers": "#0A6AC9",
    "Ottawa Senators": "#C8102E",
    "Philadelphia Flyers": "#F74902",
    "Pittsburgh Penguins": "#FCB514",
    "San Jose Sharks": "#218F97",
    "Seattle Kraken": "#29CBB2",
    "St. Louis Blues": "#002F87",
    "Tampa Bay Lightning": "#033990",
    "Toronto Maple Leafs": "#00205B",
    "Utah Mammoth": "#70D1FE",
    "Vancouver Canucks": "#02801B",
    "Vegas Golden Knights": "#D7A22F",
    "Washington Capitals": "#BE0D0D",
    "Winnipeg Jets": "#002963",
}


def load_historical_stats(position):

    # Load data
    df = pd.concat([pd.read_csv(f"../nhl-data/data/{f}") for f in os.listdir("../nhl-data/data") if position in f])

    # Filter to NHL regular seasons
    df = df[(df["gameTypeId"] == "regular season") & (df["leagueAbbrev"] == "NHL")]
    df = df.sort_values(['playerId', 'season'])

    # Clean columns
    df = df.drop(columns=["Unnamed: 0"])
    df["playerName"] = df["firstName"] + " " + df["lastName"]
    df["teamName"] = df["teamName"].str.replace("Utah Hockey Club", "Utah Mammoth", regex=False)

    # Encode avgToi to float
    def encode_avgtoi(avgtoi):
        minutes, seconds = avgtoi.split(":")
        return int(minutes) + int(seconds) / 60

    df["avgToi_adj"] = df["avgToi"].apply(encode_avgtoi)
    df["totalToi"] = df["avgToi_adj"] * df["gamesPlayed"]

    # Group trade deadline seasons together
    df = df.groupby(["playerId", "season"]).agg(
        team=("teamName", list),
        playerName=("playerName", "max"),
        age=("age", "max"),
        gamesPlayed=("gamesPlayed", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        points=("points", "sum"),
        plusMinus=("plusMinus", "sum"),
        pim=("pim", "sum"),
        powerPlayPoints=("powerPlayPoints", "sum"),
        shots=("shots", "sum"),
        totalToi=("totalToi", "sum"),
    ).reset_index()

    df["avgToi"] = df["totalToi"] / df["gamesPlayed"]
    df["yoe"] = df.groupby("playerId").cumcount() + 1
    df["playerId"] = df["playerId"].astype(str)
    df["season"] = df["season"].astype(str)
    df["final_team"] = df["team"].apply(lambda x: x[-1] if isinstance(x, list) and x else None)
    df["color"] = df["final_team"].map(TEAM_COLORS)

    # Clean season column
    season_lengths = {
        "20192020": 70,    # COVID-shortened (approx; varies a bit by team)
        "20202021": 56,    # COVID-shortened
        "20042005": 0,     # lockout — drop if present
    }

    df["seasonLength"] = df["season"].map(season_lengths).fillna(82)
    df["gpa"] = df["gamesPlayed"] / df["seasonLength"]
    df = df[df["seasonLength"] > 0]

    df["g/60"] = (df["goals"] / df["totalToi"]) * 60
    df["a/60"] = (df["assists"] / df["totalToi"]) * 60
    df["p/60"] = (df["points"] / df["totalToi"]) * 60
    df["s/60"] = (df["shots"] / df["totalToi"]) * 60
    df["ppp/60"] = (df["powerPlayPoints"] / df["totalToi"]) * 60

    return df


def load_cap_data():
    """Load player salary cap hit data
    """
    df = pd.concat([pd.read_csv(f"../nhl-data/cap_data/{f}") for f in os.listdir("../nhl-data/cap_data")])
    df = df.drop(columns=["Unnamed: 0"])
    return df

