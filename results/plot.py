import pandas as pd
import pylab as plt
from matplotlib.font_manager import FontProperties
import math

# Figure plotting
def generate_average(df):
    x = df[:1]
    y1 = df.iloc[1::3, :].mean().to_frame().T
    y2 = df.iloc[2::3, :].mean().to_frame().T
    frames = [x, y1, y2]
    return pd.concat(frames)


def load_three(attribute='', filenames=["nn-rhc-all.csv", "nn-sa-all.csv", "nn-ga-all.csv"]):
    df = pd.read_csv(filenames[0], header=None)
    df = generate_average(df)
    time_rhc = df[2:3].values.tolist()[0]
    iter_num = df[:1].values.tolist()[0]
    score_rhc = df[1:2].values.tolist()[0]

    df = pd.read_csv(filenames[1], header=None)
    df = generate_average(df)
    time_sa = df[2:3].values.tolist()[0]
    iter_num = df[:1].values.tolist()[0]
    score_sa = df[1:2].values.tolist()[0]

    df = pd.read_csv(filenames[2], header=None)
    df = generate_average(df)
    time_ga = df[2:3].values.tolist()[0]
    iter_num = df[:1].values.tolist()[0]
    score_ga = df[1:2].values.tolist()[0]
    
    if attribute == "time":
        return iter_num, time_rhc, time_sa, time_ga

    elif attribute == "score":
        return iter_num, score_rhc, score_sa, score_ga

def load_data(attribute='', filenames=["rhc-cp.csv", "sa-cp.csv", "ga-cp.csv", "mimic-cp.csv"]):
    df = pd.read_csv(filenames[0], header=None)
    df = generate_average(df)
    time_rhc = [math.log10(x) for x in df[2:3].values.tolist()[0]]
    iter_num = df[:1].values.tolist()[0]
    score_rhc = df[1:2].values.tolist()[0]

    df = pd.read_csv(filenames[1], header=None)
    df = generate_average(df)
    time_sa = [math.log10(x) for x in df[2:3].values.tolist()[0]]
    iter_num = df[:1].values.tolist()[0]
    score_sa = df[1:2].values.tolist()[0]

    df = pd.read_csv(filenames[2], header=None)
    df = generate_average(df)
    time_ga = [math.log10(x) for x in df[2:3].values.tolist()[0]]
    iter_num = df[:1].values.tolist()[0]
    score_ga = df[1:2].values.tolist()[0]

    df = pd.read_csv(filenames[3], header=None)
    df = generate_average(df)
    time_mimic = [math.log10(x) for x in df[2:3].values.tolist()[0]]
    iter_num = df[:1].values.tolist()[0]
    score_mimic = df[1:2].values.tolist()[0]
    
    if attribute == "time":
        return iter_num, time_rhc, time_sa, time_ga, time_mimic

    elif attribute == "score":
        return iter_num, score_rhc, score_sa, score_ga, score_mimic

def plot_three_time(data, title=''):
    iter_num, x_rhc, x_sa, x_ga = data
    plt.plot(iter_num, x_rhc, "r-",
             iter_num, x_sa, "g-",
             iter_num, x_ga, "b-")
    plt.xlabel("Iteration Number")
    plt.ylabel("Running Time (s)")
    plt.title(title)
    plt.legend(["RHC", "SA", "GA"], loc="best",
                fancybox=True, shadow=True,
                prop=FontProperties().set_size("small"))
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.show()

def plot_three_score(data, title=''):
    iter_num, x_rhc, x_sa, x_ga = data
    plt.plot(iter_num, x_rhc, "r-",
             iter_num, x_sa, "g-",
             iter_num, x_ga, "b-")
    plt.xlabel("Iteration Number")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(["RHC", "SA", "GA"], loc="best",
                fancybox=True, shadow=True,
                prop=FontProperties().set_size("small"))
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.show()

def plot_time(data, title=''):
    iter_num, x_rhc, x_sa, x_ga, x_mimic = data
    plt.plot(iter_num, x_rhc, "r-",
             iter_num, x_sa, "g-",
             iter_num, x_ga, "b-",
             iter_num, x_mimic, "y-")
    plt.xlabel("Iteration Number")
    plt.ylabel("Logarithmic Time")
    plt.title(title)
    plt.legend(["RHC", "SA", "GA", "MIMIC"], loc="best",
                fancybox=True, shadow=True,
                prop=FontProperties().set_size("small"))
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.show()

def plot_score(data, title=''):
    iter_num, x_rhc, x_sa, x_ga, x_mimic = data
    plt.plot(iter_num, x_rhc, "r-",
             iter_num, x_sa, "g-",
             iter_num, x_ga, "b-",
             iter_num, x_mimic, "y-")
    plt.xlabel("Iteration Number")
    plt.ylabel("Fitness Function Score")
    plt.title(title)
    plt.legend(["RHC", "SA", "GA", "MIMIC"], loc="best",
                fancybox=True, shadow=True,
                prop=FontProperties().set_size("small"))
    plt.grid(True)
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.show()

def load_single(filename):
    df = pd.read_csv(filename, header=None)
    df = generate_average(df)
    time = [math.log10(x) for x in df[2:3].values.tolist()[0]]
    x_value = df[:1].values.tolist()[0]
    score = df[1:2].values.tolist()[0]
    return x_value, time, score

def plot_single(x_value, y_value, title='', xlabel='', ylabel='', xscale='linear'):
    plt.plot(x_value, y_value, "r-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xscale == "log":
        plt.xscale(xscale, basex=10)
    plt.title(title)
    plt.grid(True)
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.show()

def main():
    """
    Continuous Peaks
    """
    #filenames=["rhc-cp.csv", "sa-cp.csv", "ga-cp.csv", "mimic-cp.csv"]
    #data_time = load_data("time", filenames)
    #data_score = load_data("score", filenames)
    #data_sa_rate = load_single("sa_rate.csv")
    #data_sa_rate_000100 = load_single("sa_rate_001-100.csv")
    #data_sa_t_101E11 = load_single("sa_t_10-1E11.csv")

    #plot_time(data_time)
    #plot_score(data_score)
    #plot_single(data_sa_rate[0], data_sa_rate[1], xlabel="Cooling Exponent", ylabel="Logarithmic Time", xscale='log')
    #plot_single(data_sa_rate[0], data_sa_rate[2], xlabel="Cooling Exponent", ylabel="Fitness Function Score", xscale='log')
    #plot_single(data_sa_rate_000100[0], data_sa_rate_000100[1], xlabel="Cooling Exponent", ylabel="Logarithmic Time")
    #plot_single(data_sa_rate_000100[0], data_sa_rate_000100[2], xlabel="Cooling Exponent", ylabel="Fitness Function Score")
    #plot_single(data_sa_t_101E11[0], data_sa_t_101E11[1], xlabel="Starting Temperature", ylabel="Logarithmic Time", xscale='log')
    #plot_single(data_sa_t_101E11[0], data_sa_t_101E11[2], xlabel="Starting Temperature", ylabel="Fitness Function Score", xscale='log')

    """
    Knapsack
    """
    #filenames=["rhc-kp.csv", "sa-kp.csv", "ga-kp.csv", "mimic-kp.csv"]
    #data_time = load_data("time", filenames)
    #data_score = load_data("score", filenames)
    #data_ga_tomate = load_single("ga-kp-population_tomate_001-100.csv")
    #data_ga_tomutate = load_single("ga-kp-population_tomutate_001-191.csv")

    #plot_time(data_time)
    #plot_score(data_score)
    #plot_single(data_ga_tomate[0], data_ga_tomate[1], xlabel="Mated Number", ylabel="Logarithmic Time")
    #plot_single(data_ga_tomate[0], data_ga_tomate[2], xlabel="Mated Number", ylabel="Fitness Function Score")
    #plot_single(data_ga_tomutate[0], data_ga_tomutate[1], xlabel="Mutated Number", ylabel="Logarithmic Time")
    #plot_single(data_ga_tomutate[0], data_ga_tomutate[2], xlabel="Mutated Number", ylabel="Fitness Function Score")


    """
    Traveling Salesman
    """
    #filenames=["rhc-ts.csv", "sa-ts.csv", "ga-ts.csv", "mimic-ts.csv"]
    #data_time = load_data("time", filenames)
    #data_score = load_data("score", filenames)
    #data_mimic_tokeep = load_single("mimic-ts_tokeep.csv")
    #data_mimic_population_tokeep = load_single("mimic-ts_population-tokeep.csv")

    #plot_time(data_time)
    #plot_score(data_score)
    #plot_single(data_mimic_tokeep[0], data_mimic_tokeep[1], xlabel="Kept Number", ylabel="Logarithmic Time")
    #plot_single(data_mimic_tokeep[0], data_mimic_tokeep[2], xlabel="Kept Number", ylabel="Fitness Function Score")
    #plot_single(data_mimic_population_tokeep[0], data_mimic_population_tokeep[1], xlabel="Population-Keep Combination", ylabel="Logarithmic Time")
    #plot_single(data_mimic_population_tokeep[0], data_mimic_population_tokeep[2], xlabel="Population-Keep Combination", ylabel="Fitness Function Score")

    """
    Neural Networks
    """
    filenames=["nn-rhc-all.csv", "nn-sa-all.csv", "nn-ga-all.csv"]
    three_time = load_three("time", filenames)
    three_score = load_three("score", filenames)
    plot_three_score(three_score)
    plot_three_time(three_time)

    nn_rhc = load_single("nn-rhc.csv")
    nn_sa = load_single("nn-sa.csv")
    plot_single(nn_rhc[0], nn_rhc[1], xlabel="Iteration Number", ylabel="Running Time (s)")
    plot_single(nn_rhc[0], nn_rhc[2], xlabel="Iteration Number", ylabel="Accuracy")
    plot_single(nn_sa[0], nn_sa[1], xlabel="Iteration Number", ylabel="Running Time (s)")
    plot_single(nn_sa[0], nn_sa[2], xlabel="Iteration Number", ylabel="Accuracy")

if __name__ == "__main__":
    main()