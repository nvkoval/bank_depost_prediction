import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def outlier_range(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_value = (Q1 - 1.5 * IQR)
    max_value = (Q3 + 1.5 * IQR)
    num_outliers = ((df[column] < min_value) | (df[column] > max_value)).sum()
    return min_value, max_value, num_outliers


def outlier_range_max(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    max_value = (Q3 + 1.5 * IQR)
    return max_value


def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    pltname = f'Normalized Distribution of Values by Category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=ax, title=pltname)

    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.margins(y=0.15)

    for container in ax.containers:
        ax.bar_label(container,
                     fmt='{:,.1f}',
                     fontsize=9,
                     rotation=90,
                     padding=3)


def bi_cat_countplot_no_outliers(df, column, hue_column):
    max_value = outlier_range_max(df, column)
    df = df[df[column] <= max_value]
    bi_cat_countplot(df, column, hue_column)


def bi_cat_countplot_sort_index(df, column, hue_column):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    pltname = f'Normalized Distribution of Values by Category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = (proportions
          .unstack(hue_column)
          .sort_index()
          .plot.bar(ax=ax, title=pltname))

    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.margins(y=0.15)

    for container in ax.containers:
        ax.bar_label(container,
                     fmt='{:,.1f}',
                     fontsize=9,
                     rotation=90,
                     padding=3)


def bi_numeric_kde(data, column):
    df0 = data[data['y'] == 'no']
    df1 = data[data['y'] == 'yes']
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df0[column], label='Not subscribed', cut=0)
    sns.kdeplot(df1[column], label='Subscribed', cut=0)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def bi_numeric_kde_no_outliers(data, column):
    max_value = outlier_range_max(data, column)
    df = data.loc[data[column] <= max_value]
    bi_numeric_kde(df, column)
