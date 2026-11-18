"""
some checks for the code
"""

import pandas as pd

from eda import (cohort_table, playtime_data, rating_data, prepare_data,
                 seven_numbers)


def small_data():
    """
    small made-up data just for tests
    """
    return pd.DataFrame({
        "appid": [1, 2, 3, 4, 5],
        "categories": ["Single-player", "Single-player;Co-op",
                       "Not-Single-player", "Single-player", "Single-player"],
        "release_date": ["2013-01-01", "2014-01-01", "2016-01-01",
                         "2018-01-01", "2019-01-01"],
        "achievements": [0, 10, 5, 20, None],
        "median_playtime": [10, 0, 40, 30, 50],
        "positive_ratings": [3, 80, 90, 0, 5],
        "negative_ratings": [1, 20, 10, 0, 5]
    })


def test_exact_label():
    """
    check the category label
    """
    df = prepare_data(small_data())
    assert df["single_player"].tolist() == [True, True, False, True, True]


def test_rating_calculation():
    """
    Check rating numbers
    """
    df = prepare_data(small_data())
    assert df.loc[0, "positive_proportion"] == 0.75
    assert pd.isna(df.loc[3, "positive_proportion"])


def test_separate_samples():
    """
    check which games stay in each group
    """
    df = prepare_data(small_data())
    single_data = df[df["single_player"]]
    play = playtime_data(single_data)
    ratings = rating_data(single_data)
    achievement_play = play[play["achievements"] > 0]
    assert play["appid"].tolist() == [1, 4]
    assert ratings["appid"].tolist() == [2]
    assert achievement_play["appid"].tolist() == [4]


def test_missing_values():
    """
    check missing stuff
    """
    games = small_data()
    games["achievements"] = [None, 10, 5, 20, None]
    games["median_playtime"] = [None, 0, 40, 30, 50]
    df = prepare_data(games)
    assert pd.isna(df.loc[0, "achievements"])
    assert pd.isna(df.loc[0, "median_playtime"])
    assert pd.isna(df.loc[4, "has_achievements"])


def test_release_cohort_boundaries():
    """
    Check year groups
    """
    df = prepare_data(small_data())
    single_data = df[df["single_player"]]
    table = cohort_table(single_data)
    assert table[False].tolist() == [1, 0, 0]
    assert table[True].tolist() == [0, 1, 1]
    assert pd.isna(df.loc[4, "release_cohort"])


def test_seven_number_summary():
    """
    check the summary numbers
    """
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = seven_numbers(df, ["x"])
    assert result.loc["x", "count"] == 5
    assert result.loc["x", "mean"] == 3
    assert abs(result.loc["x", "std"] - 2.5 ** 0.5) < 0.000001
    assert result.loc["x", "min"] == 1
    assert result.loc["x", "25%"] == 2
    assert result.loc["x", "50%"] == 3
    assert result.loc["x", "75%"] == 4
    assert result.loc["x", "max"] == 5


def main():
    """
    Run tests
    """
    test_exact_label()
    test_rating_calculation()
    test_separate_samples()
    test_missing_values()
    test_release_cohort_boundaries()
    test_seven_number_summary()
    print("All 6 tests passed.")


if __name__ == "__main__":
    main()
