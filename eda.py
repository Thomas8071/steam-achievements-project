"""
steam games analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COHORTS = ["2010-2013", "2014-2016", "2017-2018"]


def prepare_data(games):
    """
    clean data and add some columns
    """
    df = games.copy()

    single_player = []
    for category in df["categories"]:
        if pd.isna(category):
            single_player.append(False)
        elif "Single-player" in category.split(";"):
            single_player.append(True)
        else:
            single_player.append(False)
    df["single_player"] = single_player

    has_achievements = []
    for count in df["achievements"]:
        if pd.isna(count):
            has_achievements.append(pd.NA)
        elif count > 0:
            has_achievements.append(True)
        else:
            has_achievements.append(False)
    df["has_achievements"] = has_achievements

    df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
    proportions = []
    for index in df.index:
        total = df.loc[index, "total_ratings"]
        positive = df.loc[index, "positive_ratings"]
        if pd.isna(total) or total == 0:
            proportions.append(np.nan)
        else:
            proportions.append(positive / total)
    df["positive_proportion"] = proportions

    dates = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = dates.dt.year
    cohorts = []
    for year in df["release_year"]:
        if 2010 <= year <= 2013:
            cohorts.append("2010-2013")
        elif 2014 <= year <= 2016:
            cohorts.append("2014-2016")
        elif 2017 <= year <= 2018:
            cohorts.append("2017-2018")
        else:
            cohorts.append(np.nan)
    df["release_cohort"] = cohorts
    return df


def playtime_data(single_data):
    """
    games used for playtime
    """
    play = single_data[single_data["median_playtime"] > 0].copy()
    play = play[play["achievements"].notna()].copy()
    return play


def rating_data(single_data):
    """
    games used for ratings
    """
    rating_df = single_data[single_data["achievements"] > 0].copy()
    rating_df = rating_df[rating_df["total_ratings"] >= 100].copy()
    rating_df = rating_df[rating_df["positive_proportion"].notna()].copy()
    return rating_df


def seven_numbers(df, columns):
    """
    some stats for each column
    """
    rows = []
    for column in columns:
        values = df[column]
        row = [values.count(), values.mean(), values.std(), values.min(),
               values.quantile(0.25), values.median(),
               values.quantile(0.75), values.max()]
        rows.append(row)

    names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return pd.DataFrame(rows, index=columns, columns=names)


def cohort_table(df):
    """
    count games by year group
    """
    rows = []
    for years in COHORTS:
        group = df[df["release_cohort"] == years]
        without = group[group["achievements"] == 0]
        with_achievements = group[group["achievements"] > 0]
        rows.append([len(without), len(with_achievements)])
    return pd.DataFrame(rows, index=COHORTS, columns=[False, True])


def question1(single_data, folder):
    """
    compare playtime with and without achievements
    """
    play = playtime_data(single_data)
    without = play[play["achievements"] == 0]
    with_achievements = play[play["achievements"] > 0]
    without_log = np.log10(without["median_playtime"])
    with_log = np.log10(with_achievements["median_playtime"])

    print("\nQuestion 1")
    print("Games without achievements:", len(without))
    print("Games with achievements:", len(with_achievements))
    print("Median minutes without achievements:",
          without["median_playtime"].median())
    print("Median minutes with achievements:",
          with_achievements["median_playtime"].median())

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.boxplot([without_log, with_log],
               tick_labels=["No achievements", "Has achievements"],
               flierprops={"markersize": 2, "alpha": 0.2})
    ax.set_title("playtime")
    ax.set_ylabel("playtime (log10 minutes)")
    fig.tight_layout()
    fig.savefig(folder / "figure1.png", dpi=180)
    plt.close(fig)


def question2(single_data, folder):
    """
    achievement count and playtime or ratings
    """
    play = playtime_data(single_data)
    play = play[play["achievements"] > 0].copy()
    rating_df = rating_data(single_data)
    play_result = stats.spearmanr(
        play["achievements"], play["median_playtime"])
    rating_result = stats.spearmanr(
        rating_df["achievements"], rating_df["positive_proportion"])

    print("\nQuestion 2")
    print("Playtime sample:", len(play))
    print("Rating sample:", len(rating_df))
    print("Correlation with playtime:", play_result.statistic)
    print("Correlation with ratings:", rating_result.statistic)
    print("Different achievement counts in playtime data:",
          play["achievements"].nunique())
    print("Different playtime values:", play["median_playtime"].nunique())
    print("Different achievement counts in rating data:",
          rating_df["achievements"].nunique())
    print("Different rating proportions:",
          rating_df["positive_proportion"].nunique())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))

    axes[0].scatter(np.log10(play["achievements"]),
                    np.log10(play["median_playtime"]), s=7, alpha=0.18)
    axes[0].set_title("achievements vs time")
    axes[0].set_xlabel("Achievements (log10)")
    axes[0].set_ylabel("Playtime (log10 minutes)")

    axes[1].scatter(np.log10(rating_df["achievements"]),
                    rating_df["positive_proportion"], s=7, alpha=0.18)
    axes[1].set_title("Ratings")
    axes[1].set_xlabel("Achievements (log10)")
    axes[1].set_ylabel("Positive rating proportion")
    fig.tight_layout()
    fig.savefig(folder / "figure2.png", dpi=180)
    plt.close(fig)


def question3(single_data, folder):
    """
    compare the release year groups
    """
    table = cohort_table(single_data)
    print("\nQuestion 3: False = no achievements; True = achievements")
    print(table.to_string())
    expected = stats.contingency.expected_freq(table)
    print("Smallest expected count:", expected.min())
    percentages = []
    for years in COHORTS:
        without_count = table.loc[years, False]
        with_count = table.loc[years, True]
        total = without_count + with_count
        percentages.append(with_count / total * 100)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.bar(COHORTS, percentages)
    for index in range(len(bars)):
        bar = bars[index]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height() + 2
        ax.text(x, y, f"{percentages[index]:.1f}%", ha="center")
    ax.set_title("games by year")
    ax.set_xlabel("release Year")
    ax.set_ylabel("Games with achievements (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(folder / "figure3.png", dpi=180)
    plt.close(fig)


def main():
    """
    Read file and run
    """
    folder = Path(__file__).resolve().parent
    games = pd.read_csv(folder / "steam.csv")
    df = prepare_data(games)
    single_data = df[df["single_player"]].copy()
    missing = games.isna().sum()
    print("\nMissing values per column:")
    print(missing.to_string())
    numeric = ["achievements", "median_playtime", "positive_ratings",
               "negative_ratings", "total_ratings", "positive_proportion",
               "release_year"]
    print("\nNumerical summaries for single-player games:")
    print(seven_numbers(single_data, numeric).round(3).to_string())
    print("\nGames with and without single-player support:")
    print("True:", len(single_data))
    print("False:", len(games) - len(single_data))

    with_achievements = single_data[single_data["achievements"] > 0]
    without = single_data[single_data["achievements"] == 0]
    print("\nAchievement availability in single-player games:")
    print("True:", len(with_achievements))
    print("False:", len(without))
    print("Original rows:", len(games))
    print("Original columns:", len(games.columns))
    print("Rows with missing values:", len(games) - len(games.dropna()))
    print("Missing cells:", missing.sum())
    print("Repeated app IDs:", games["appid"].duplicated().sum())
    print("Zero playtime games:",
          len(single_data[single_data["median_playtime"] == 0]))
    print("Missing release years:", single_data["release_year"].isna().sum())

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})
    question1(single_data, folder)
    question2(single_data, folder)
    question3(single_data, folder)


if __name__ == "__main__":
    main()
