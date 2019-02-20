import pandas as pd
import os
import re
import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
from datetime import datetime





class smartband_preprocess():
    column_names = ['sensor_addr', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z',
                    'mag_x', 'mag_y', 'mag_z', 'pressure', 'temperature', 'force', 'quat_1', 'quat_2', 'quat_3',
                    'quat_4']
    def __init__(self):
        return

    def decode_filename(self, name):
        # date time
        match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}', name)

        if (match):
            DT = datetime.strptime(match.group(), '%Y-%m-%d %H-%M-%S')
        else:
            DT = None

        # fine S T R number in file name
        pattern = r'S(\d*)'
        S = re.findall(pattern, name)[0]
        pattern = r'T(\d*)'
        T = re.findall(pattern, name)[0]
        pattern = r'R(\d*)'
        R = re.findall(pattern, name)[0]
        pattern = r'R\d* (.+).csv'
        comment = re.findall(pattern, name)
        if (comment):
            comment = comment[0]
        else:
            comment = ''
        return DT, S, T, R, comment



    def calibrate_loadcell(self, df):
        """
        corret offset
        :param df:
        :return:
        """
        min = np.min(df["force"][5:1000])
        df["force"] = df["force"] - min
        return df

    def resample(self, raw_dir='data/', output_dir='resampled/', interpolate='linear', res_freq="10ms", print_info = False):
        """
        :param raw_dir:
        :param output_dir:
        :param interpolate:
        :param res_freq:
        :return:
        """
        files= os.listdir(raw_dir)
        for file in files:
            if not os.path.isdir(file):
                if print_info:
                    print("Processing", file)

                df = pd.read_csv(raw_dir+"/"+file, names=self.column_names, header=0)
                sensor_addrs = df['sensor_addr'].unique()
                temp_df = pd.DataFrame()

                for addr in sensor_addrs:
                    sensor_df = df[df['sensor_addr'] == addr].copy()
                    # do calibration
                    sensor_df = self.calibrate_loadcell(sensor_df)

                    # resampling only support time indexing.
                    sensor_df['time_index'] = pd.to_datetime(sensor_df['timestamp'],  unit='ms')
                    sensor_df = sensor_df.set_index('time_index')
                    sensor_df = sensor_df.resample(res_freq).mean()
                    #sensor_df = sensor_df[sensor_df.columns.difference(['timestamp'])].resample(res_freq).mean()
                    # fill the NaN data
                    if(interpolate is not None):
                        sensor_df = sensor_df.interpolate(method=interpolate)
                    # recover data
                    sensor_df['sensor_addr'] = addr
                    sensor_df['timestamp'] = sensor_df['timestamp'].round(0)
                    sensor_df = sensor_df[self.column_names]                    # recover column order

                    temp_df = temp_df.append(sensor_df, ignore_index=True)

                temp_df.to_csv(output_dir+"/"+file , index=False, float_format = '%.7f')


    def linear_acc(self, df):
        """
        v -> x = 2 * (q -> x*q -> z - q -> w*q -> y);
        v -> y = 2 * (q -> w*q -> x + q -> y*q -> z);
        v -> z = q -> w*q -> w - q -> x*q -> x - q -> y*q -> y + q -> z*q -> z;
        """
        df["grav_x"] = 2* (df["quat_2"] * df["quat_4"] - df["quat_1"] * df["quat_3"])
        df["grav_y"] = 2* (df["quat_1"] * df["quat_2"] + df["quat_3"] * df["quat_4"])
        df["grav_z"] = df["quat_1"]*df["quat_1"] - df["quat_2"]*df["quat_2"] - df["quat_3"]*df["quat_3"] + df["quat_4"]*df["quat_4"]

        df["ln_acc_x"] = (df["acc_x"] - df["grav_x"])*9.81
        df["ln_acc_y"] = (df["acc_y"] - df["grav_y"])*9.81
        df["ln_acc_z"] = (df["acc_z"] - df["grav_z"])*9.81

        df["quantized_acc"] = np.sqrt(df["ln_acc_x"]**2 + df["ln_acc_y"]**2 + df["ln_acc_z"]**2)

        return df

    def rotate_gyro(self, df):
        df["quantized_rotate"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
        return  df


    def normalize_sensor_data(self, df):

        df['quantized_acc_x'] = (df["acc_x"] * 4).clip(-128, 127).round(0)
        df['quantized_acc_y'] = (df["acc_y"] * 4).clip(-128, 127).round(0)
        df['quantized_acc_z'] = (df["acc_z"] * 4).clip(-128, 127).round(0)

        df['quantized_gyro_x'] = (df["gyro_x"] / 4).clip(-128, 127).round(0)
        df['quantized_gyro_y'] = (df["gyro_y"] / 4).clip(-128, 127).round(0)
        df['quantized_gyro_z'] = (df["gyro_z"] / 4).clip(-128, 127).round(0)

        # test
        df['quantized_acc'] = (df['quantized_acc'] * 4).clip(-128, 127).round(0)
        df['quantized_rotate'] = (df['quantized_rotate'] / 4).clip(-128, 127).round(0)
        df['quantized_force'] = (df['force'] * 2).clip(-128, 127).round(0)   # 0-50N to 0 to +128
        return df
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def lowpass(self, df, sampling_rate, cutoff):
        for column in df.columns:
            if(df[column].dtypes == np.float64 or df[column].dtypes == np.int64):
                df[column] = self.butter_lowpass_filter(df[column].values, cutoff, sampling_rate, order=2)
        return df

    # this method is used specificatly for 1 or more sensor data merge.
    # method = 'all', keep all the source data;
    # method = 'mid , cut to first and last length 'cut''
    def generate_merged_file(self, dir_path, output_file_path="testdata.csv", method='all', sampling_rate=100, cut=500, lowpass_cutoff=10,
                             show_info=True, normalize=False, keep_index=False):
        """
        :param dir_path:
        :param output_file_path:
        :param method:
        :param cut:         cut the frames at beginning and ending.
        :param lowpass:
        :param show_info:
        :param normalize:
        :param keep_index:
        :return:
        """
        output_dataframe = pd.DataFrame()
        files= os.listdir(dir_path)
        for file in files:
            if not os.path.isdir(file):
                f = open(dir_path+"/"+file)
                if show_info:
                    print("file found:", f.name)
                name = f.name
                f.close()

                # fine S T R number in file name
                DT, S, T, R, comment = self.decode_filename(name)

                # get the data itself
                df = pd.read_csv(dir_path+"/"+file, header=0, names=self.column_names)  # replace the header

                sensor_addrs = df['sensor_addr'].unique()
                temp_df = pd.DataFrame()

                for addr in sensor_addrs:
                    sensor_df = df[df['sensor_addr'] == addr]
                    #sensor_df = sensor_df.reset_index()

                    if (method == 'mid'):
                        if (sensor_df.size > 2 * cut):
                            sensor_df = sensor_df.loc[cut:sensor_df.size - cut]
                        else:
                            print("the source data is not long enough(", sensor_df.size, "), please reduce the 'cut'(",
                                  cut, ')')
                    elif method == 'front':
                        if (sensor_df.size >  cut):
                            sensor_df = sensor_df.loc[cut:sensor_df.size]
                        else:
                            print("the source data is not long enough(", sensor_df.size, "), please reduce the 'cut'(",
                                  cut, ')')
                    else:
                        if (method != 'all'):
                            print("invalid parameters:", method)
                    temp_df = temp_df.append(sensor_df, ignore_index=True)

                # calculate gravity vector and linear acceleration
                temp_df = self.linear_acc(temp_df)
                # calculate rotation Scalar: rotate = sqrt(gyrox2 + gyroy2 + gyroz2)
                temp_df = self.rotate_gyro(temp_df)

                if(lowpass_cutoff != 0):
                    temp_df = self.lowpass(temp_df, sampling_rate, lowpass_cutoff)
                if (normalize):
                    temp_df = self.normalize_sensor_data(temp_df)

                temp_df['S'] = S  # add subject number
                temp_df['T'] = T  # add test number
                temp_df['R'] = R  # add repetition number
                if(comment):
                    temp_df['comment'] = comment
                else:
                    temp_df['comment'] = ''

                output_dataframe = output_dataframe.append(temp_df, ignore_index=True)
                if show_info:
                    print("info:", 'S',S, ' T', T, ' R', R, ' Comment:', comment)

        output_dataframe.to_csv(output_file_path, index=keep_index)

        if show_info:
            print(output_dataframe.info())
            print("data save to: ", output_file_path)
        return

    def plot(self, data_dir='resampled', fig_dir='figs', rankedby='S', datalist = None):
        import matplotlib.pyplot as plt
        if datalist == None:
            return
        files= os.listdir(data_dir)
        for file in files:
            if not os.path.isdir(file):
                file_name = data_dir+"/"+file
                # fine S T R number in file name
                DT, S, T, R, comment = self.decode_filename(file_name)
                # get the data itself
                df = pd.read_csv(file_name, header=0, names=self.column_names)  # replace he header

                # get more data if needed
                df = self.linear_acc(df)
                df = self.rotate_gyro(df)
                df = self.normalize_sensor_data(df)

                fig, ax = plt.subplots(1)
                for column in datalist:
                    ax.plot(df[column])
                ax.set_ylim((None,128))
                plt.legend()

                encode_title = "S" + S + ' T' + T + ' R' + R +' '+ comment
                if(rankedby == 'T'):
                    encode_title = 'T' + T + " S" + S + ' R' + R +' '+ comment
                elif rankedby == 'R':
                    encode_title = 'R' + R + " S" + S + ' T' + T +' '+ comment

                plt.title(encode_title)
                plt.savefig(fig_dir + "/" + encode_title + '.png')
                plt.close(fig)


if __name__ == "__main__":
    data = smartband_preprocess()

    # generate resampling data
    data.resample(interpolate='index', res_freq="10ms", print_info= True)

    data.plot(rankedby='T', datalist=['quantized_force', 'quantized_acc', 'quantized_rotate'])