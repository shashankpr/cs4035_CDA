import pandas as pd
import numpy as np
import logging
from collections import Counter

import preprocess

logging.basicConfig(level=logging.DEBUG)


def aggr_src_ip(df):
    src_ip = df.groupby(['SrcAddr'])
    logging.debug(src_ip.size())

    return src_ip


def get_num_of_Sport(src_ip):
    num_of_src_ports = src_ip['Sport'].count().get_values()

    return num_of_src_ports


def get_num_of_Dport(src_ip):
    num_of_dest_ports = src_ip['Dport'].count().get_values()

    return num_of_dest_ports


def get_num_of_dAddr(src_ip):
    num_of_dest_addr = src_ip['DstAddr'].count().get_values()

    return num_of_dest_addr


def get_num_of_unique_Sports(src_ip):
    unique_src_ports = src_ip['Sport'].unique()
    num_unique_src_ports = []
    for ports in unique_src_ports.get_values():
        num_unique_src_ports.append(ports.size)

    return np.asarray(num_unique_src_ports)


def get_num_of_unique_Dports(src_ip):
    unique_dest_ports = src_ip['Dport'].unique()
    num_unique_dest_ports = []
    for ports in unique_dest_ports.get_values():
        num_unique_dest_ports.append(ports.size)

    return np.asarray(num_unique_dest_ports)


def get_num_of_unique_dAddr(src_ip):
    unique_dest_addr = src_ip['DstAddr'].unique()
    num_unique_dest_addr = []
    for ports in unique_dest_addr.get_values():
        num_unique_dest_addr.append(ports.size)

    return np.asarray(num_unique_dest_addr)


def get_packet_features(src_ip):
    # Get mean Duration
    mean_duration = src_ip['Dur'].mean().get_values()

    # Get mean Total Packets
    mean_packets = src_ip['TotPkts'].mean().get_values()

    # Get mean TotBytes
    mean_bytes = src_ip['TotBytes'].mean().get_values()
    # logging.debug("bytes dim : {}".format(mean_bytes.shape))

    # Get mean SrcBytes
    mean_src_bytes = src_ip['SrcBytes'].mean().get_values()
    # logging.debug("Src bytes dim : {}".format(mean_src_bytes.shape))

    return [np.asarray(mean_duration), np.asarray(mean_packets), np.asarray(mean_bytes), np.asarray(mean_src_bytes)]


def build_training_set(df):
    src_ip = aggr_src_ip(df)

    num_src_ports = get_num_of_Sport(src_ip)
    num_dest_ports = get_num_of_Dport(src_ip)
    num_dest_addr = get_num_of_dAddr(src_ip)
    num_unique_src_ports = get_num_of_unique_Sports(src_ip)
    num_unique_dest_ports = get_num_of_unique_Dports(src_ip)
    num_unique_dest_addr = get_num_of_unique_dAddr(src_ip)
    mean_duration = get_packet_features(src_ip)[0]
    mean_packets = get_packet_features(src_ip)[1]
    mean_bytes = get_packet_features(src_ip)[2]
    mean_src_bytes = get_packet_features(src_ip)[3]

    src_ips = df['SrcAddr'].unique().sort()

    logging.debug(num_src_ports.shape)
    logging.debug(num_dest_ports.shape)
    logging.debug(num_dest_addr.shape)
    logging.debug(num_unique_src_ports.shape)
    logging.debug(num_unique_dest_ports.shape)
    logging.debug(num_unique_dest_addr.shape)
    logging.debug(mean_duration.shape)
    logging.debug(mean_packets.shape)
    logging.debug(mean_bytes.shape)
    logging.debug(mean_src_bytes.shape)

    feat = {}
    feat['Src_IPs'] = src_ips
    feat['#SPorts'] = num_src_ports
    feat['#DPorts'] = num_dest_ports
    feat['#DAddr'] = num_unique_dest_addr
    feat['#USPorts'] = num_unique_src_ports
    feat['#UDPorts'] = num_unique_dest_ports
    feat['#UDAddr'] = num_unique_dest_addr
    feat['Mean_Dur'] = mean_duration
    feat['Mean_Bytes'] = mean_bytes
    feat['Mean_Packets'] = mean_packets
    feat['Mean_SRC_Bytes'] = mean_src_bytes

    # matrix = np.matrix([src_ips, num_src_ports, num_dest_ports, num_dest_addr, num_unique_src_ports
    #                        , num_unique_dest_ports, num_unique_dest_addr, mean_duration, mean_packets,
    #                     mean_bytes, mean_src_bytes])

    labels = np.asarray(df['Label'])

    all_labels = src_ip['Label'].unique()
    new_labels = []
    for labs in all_labels.get_values():
        if 1 in labs:
            new_labels.append(1)
        else:
            new_labels.append(0)

    new_labels = np.asarray(new_labels)
    features = pd.DataFrame(feat)

    logging.debug("Feature shape : {}".format(features.shape))
    features.info()
    logging.debug("Label Shape : {}".format(new_labels.shape))
    logging.debug("Labels : {}".format(new_labels))
    logging.debug("Labels composition : {}".format(Counter(new_labels)))
    return features, new_labels


# if __name__ == '__main__':
#     all_files = preprocess.load_data()
#     training_df = preprocess.store_to_dataframe(all_files)
#     processed_df = preprocess.preprocess_labels(training_df)
#     features, labels = build_training_set(processed_df)
