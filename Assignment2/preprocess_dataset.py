import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_excel():
    NORMAL_EXCEL = 'dataset/SWaT_Dataset_Normal_v0.xlsx'
    NOR_ATT_EXCEL = 'dataset/SWaT_Dataset_Attack_v0.xlsx'

    normal_file = pd.read_excel(NORMAL_EXCEL, header=1, index_col=None)
    attack_file = pd.read_excel(NOR_ATT_EXCEL, header=1, index_col=None)

    # Clean the column names in Attack file
    new_cols = map(lambda x : x.lstrip(), attack_file.columns.values)
    attack_file.columns = new_cols

    # Clean the 'Timestamp' of Normal file
    normal_file.rename(columns={' Timestamp': 'Timestamp'}, inplace=True)

    # Clean the labels in 'Normal/Attack' column -> 'A ttack' - 'Attack'
    attack_file['Normal/Attack'].replace('A ttack', 'Attack', inplace=True)

    return normal_file, attack_file


def excel_to_csv():
    normal_file, attack_file = process_excel()
    normal_file.to_csv("dataset/Sensor_data_Normal.csv", index=None)
    attack_file.to_csv("dataset/Sensor_data_NA.csv", index=None)

if __name__ == '__main__':
    excel_to_csv()
