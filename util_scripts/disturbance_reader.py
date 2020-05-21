import csv
import casadi as ca
import pandas as pd
import numpy as np

# default='warn'
pd.options.mode.chained_assignment = None

# TODO: add to the logging the type and the gain of the dataset used for the scenario
# TODO: move settings of the simulation into the json file with the same name instead of into the file name


class Disturbance:

    def __init__(self, disturbance_config):
        """
        Construct the disturbance class by providing a configuration dictionary.
         Example configuration:
        config = {
            "disturbance_data_name": "data/disturbance_data/hour_poop_rain.csv",
            "use_rain": True,
            "use_poop": True,
            "rain_gain": 13,
            "poop_gain": 10,
        }

        :param disturbance_config: A dictionary with the configuration
        """

        # Use pandas read_csv to load the dataset
        self.disturbance_df = pd.read_csv(disturbance_config["disturbance_data_name"], header=None,
                                          names=["Hour", "Poop", "Rain"])

        # It doesnt make sense to make another column with the hours that are equivalent to the indices,
        # we drop that row only in the memory loaded version of the df
        self.disturbance_df = self.disturbance_df.drop(["Hour"], axis=1)

        self.config = disturbance_config

        self.use_rain = self.config["use_rain"]
        self.use_poop = self.config["use_poop"]
        self.rain_gain = self.config["rain_gain"]
        self.poop_gain = self.config["poop_gain"]

    def get_k_poop_disturbance(self, k):
        """
        Get the k-th poop disturbance
        :param k: k-th index in the simulation
        :return: k-th index of the poop disturbance as a float
        """

        k_poop_disturbance = self.disturbance_df["Poop"].iloc[k]
        return k_poop_disturbance

    def get_k_rain_disturbance(self, k):
        """
        Get the k-th rain disturbance
        :param k: k-th index in the simulation
        :return: k-th index of the rain disturbance as a float
        """

        k_rain_disturbance = self.disturbance_df["Rain"].iloc[k]
        return k_rain_disturbance

    def get_disturbance_df(self):
        """
        Get the whole disturbance DataFrame
        :return: Disturbance DataFrame
        """
        return self.disturbance_df

    def get_pred_horizon_df(self, k, pred_horizon):
        """
        Slices the disturbance DataFrame according to the k-th index and to the length of the prediction horizon. Note,
        that if the k-th index is - 1, a zero padded row is added to the top assuming that the disturbance before time\
         0, is 0. Furthermore, if the horizon is over the length of the DataFrame datapoints, zero padding is added to
         fill the return slice to the length of the horizon.
        :param k: k-th index of the simulation
        :param pred_horizon: Length of the prediction horizon
        :return: Slice of the disturbance DataFrame
        """

        # print(f"K {k}")
        num_df_rows = self.disturbance_df.shape[0]

        if k == - 1:
            # Make sure that the slice from the DataFrame is one shorter. This is done to
            # allow for a insertion of a zero row as the very first element, since before time 0, it is assumed
            # that the disturbance 0.

            k = 0
            pred_horizon = pred_horizon - 1
            rows_k_to_pred_horizon = self.disturbance_df.iloc[k:k + pred_horizon]

            # Create empty DataFrame of 1 line with zeros and then append the sliced disturbance rows to the bottom
            zero_line_df = pd.DataFrame(columns=["Poop", "Rain"])
            zero_line_df.loc[0] = [0, 0]
            rows_k_to_pred_horizon = zero_line_df.append(rows_k_to_pred_horizon).reset_index(drop=True)

            return rows_k_to_pred_horizon

        elif k + pred_horizon < num_df_rows:
            # This happens when the k-th index is comfortably in the middle of the DataFrame and the horizon doesn't
            # point outside of the DataFrame.

            rows_k_to_pred_horizon = self.disturbance_df.iloc[k:k + pred_horizon]
            # print(rows_k_to_pred_horizon)
            return rows_k_to_pred_horizon

        elif k > num_df_rows:
            # If for some reason the k goes over the number of data-points we have, the prediction horizon will be
            # filled with only zeros.

            num_missing_rows = pred_horizon
            rows_k_to_pred_horizon = pd.DataFrame(columns=["Poop", "Rain"], data=np.zeros((num_missing_rows, 2)))

            return rows_k_to_pred_horizon

        else:
            # This happens when the horizon points to the outside of the DataFrame, but the k-th index is still within

            rows_k_to_pred_horizon = self.disturbance_df.iloc[k:-1]
            num_slice_row = rows_k_to_pred_horizon.shape[0]

            # Create missing entries and append to the series sliced to have the length of the desired horizon
            num_missing_rows = pred_horizon - num_slice_row

            # Create new Df and fill it with zeros with the length of the missing rows from the horizon
            zero_df = pd.DataFrame(columns=["Poop", "Rain"], data=np.zeros((num_missing_rows, 2)))

            # Append to rows_k_to_pred horizon, and then reset the index. This is done because pandas by
            # default would the keep the old indices from the slice above.
            # print(rows_k_to_pred_horizon)
            rows_k_to_pred_horizon = rows_k_to_pred_horizon.append(zero_df).reset_index(drop=True)

            # print(type(rows_k_to_pred_horizon))

            return rows_k_to_pred_horizon

    def get_k_disturbance(self, k, pred_horizon):
        """
        Returns the corresponding disturbance to the k-th index and to the end of the prediction horizon.
        :param k: k-th index in the simulation
        :param pred_horizon: Length of the prediction horizon
        :return: Returns the disturbance as a weighted sum of the two possibilities, either Rain or Poop. The return
         is of type ca.DM vector.
        """

        rows_k_to_pred_horizon = self.get_pred_horizon_df(k, pred_horizon)

        if self.use_poop is True and self.use_rain is False:
            # We ONLY want to use poop data
            rows_k_to_pred_horizon["Combined"] = rows_k_to_pred_horizon.loc[:, "Poop"] * self.poop_gain

        elif self.use_poop is True and self.use_rain is True:
            # We want to use BOTH poop and rain data
            # rows_k_to_pred_horizon["Rain"] = rows_k_to_pred_horizon["Rain"].multiply(self.rain_gain)

            # df.loc[:,'quantity'] *= -1
            rows_k_to_pred_horizon.loc[:, "Rain"] *= self.rain_gain
            rows_k_to_pred_horizon.loc[:, "Poop"] *= self.rain_gain

            rows_k_to_pred_horizon["Combined"] = rows_k_to_pred_horizon["Rain"] + rows_k_to_pred_horizon["Poop"]

        elif self.use_poop is False and self.use_rain is True:
            # We ONLY want to use rain data
            rows_k_to_pred_horizon["Combined"] = rows_k_to_pred_horizon["Rain"] * self.rain_gain

        else:
            # We don't want to use either of the disturbances
            # Obviously this doesnt much make sense in our setup, but just in case, it returns
            # a vector of zeros for the entire prediction horizon
            rows_k_to_pred_horizon["Combined"] = rows_k_to_pred_horizon["Rain"] * 0

        # create ca.DM from pandas df
        combined_disturbance = rows_k_to_pred_horizon["Combined"].tolist()

        # print(combined_disturbance)
        # print(len(combined_disturbance))
        return ca.DM(combined_disturbance)

    def get_k_delta_disturbance(self, k, pred_horizon):
        """
        Returns the corresponding disturbance to the k-th and the k-1th index and to the end of the prediction horizon.
         The corresponding math is this: # dU_d_k = U_d_k - U_d_k - 1
        :param k: k-th index in the simulation
        :param pred_horizon: Length of the prediction horizon
        :return: Returns the difference between two of the disturbance horizons. Keeps the ca.DM vector type
        """

        k_prev_to_horizon_disturb = self.get_k_disturbance(k - 1, pred_horizon)
        k_to_horizon_disturb = self.get_k_disturbance(k, pred_horizon)

        df3 = k_to_horizon_disturb - k_prev_to_horizon_disturb

        return df3
