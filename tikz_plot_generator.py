import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_df():

    network_df = pd.read_pickle("data/mpc_data/05_27_20/mpc_simulation_df_00_45_24_EULER.pkl")
    return network_df


def plot_df(network_df):

    # ON OFF CONTROLLER
    # pump1/2 current_setting
    # t1/t2 depth, flooding
    # disturbances, same as the flow of the pumps

    fig, axes = plt.subplots(nrows=4, ncols=1)

    network_df["pump1_current_setting"].plot(ax=axes[0])
    network_df["pump2_current_setting"].plot(ax=axes[0])
    axes[0].set_title("Pump current settings [CMS]")
    axes[0].legend(["Pump 1", "Pump 2"])
    axes[0].grid()


    network_df["tank1_depth"].plot(ax=axes[1])
    network_df["tank2_depth"].plot(ax=axes[1])
    axes[1].set_title("Tank depth [m]")
    axes[1].legend(["Tank 1", "Tank 2"])
    axes[1].grid()

    network_df["tank1_flooding"].plot(ax=axes[2])
    network_df["tank2_flooding"].plot(ax=axes[2])
    axes[2].set_title("Tank flooding [m]")
    axes[2].legend(["Tank 1", "Tank 2"])
    axes[2].grid()
    # axes[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%1'))

    network_df["disturbance"].plot(ax=axes[3])
    axes[3].set_title("Combined disturbance [CMS]")
    axes[3].legend(["Disturbance"])
    axes[3].grid()

    plt.savefig("data/figures/final_local_model.png")
    plt.show()



if __name__ == "__main__":
    network_df = load_df()

    plot_df(network_df)




