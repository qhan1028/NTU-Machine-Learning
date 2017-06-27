# ML 2017 final
# DengAI: Predicting Disease Spread

import numpy as np


def read_features(filename):
    
    def parse(x):
        if x != '' and x != '\n':
            return float(x)
        else:
            return np.inf

    data_sj, data_iq = [], []
    index_sj, index_iq = [], []
    week_sj, week_iq = [], []

    #select_features = [0, 1, 2, 4, 10, 14, 16, 19, 0, 0]
    #select_features = [0, 1, 2, 4, 7, 10, 13, 14, 15, 16, 18, 19]
    select_features = [7, 13, 15, 18] + [0] * 2
    '''
    7:  reanalysis_dew_point_temp_k
    13: reanalysis_specific_humidity_g_per_kg
    15: station_avg_temp_c
    18: station_min_temp_c
    '''

    with open(filename, 'r') as f:
        f.readline()

        for line in f:
            city, year, weekofyear, weekstartdate, *features = line.split(',')

            features = np.array(features)[select_features]
            data = [parse(x) for x in features]
            data[-1] = 1. if int(weekofyear) > 30 else 0
            data[-2] = data[0] * data[1]

            if city == 'sj':
                data_sj.append( data )
                index_sj.append( [city, year, weekofyear] )
                week_sj.append( int(weekofyear) )
            elif city == 'iq':
                data_iq.append( data )
                index_iq.append( [city, year, weekofyear] )
                week_iq.append( int(weekofyear) )
            else:
                print('unknown city')

    data2_sj, data2_iq = [], []
    
    (past, future) = (9, 2)
    data_sj = [data_sj[0]] * past + data_sj + [data_sj[-1]] * future
    for i in range(len(data_sj)):

        if i >= past and i < len(data_sj)-future:
            data = list( np.reshape( data_sj[i-past : i+1+future], -1) )
            data2_sj.append( data )
    
    (past, future) = (3, 1)
    data_iq = [data_iq[0]] * past + data_iq + [data_iq[-1]] * future
    for i in range(len(data_iq)):

        if i >= past and i < len(data_iq)-future:
            data = list( np.reshape( data_iq[i-past : i+1+future], -1) )
            data2_iq.append( data )
 
    data2_sj, data2_iq = np.array(data2_sj), np.array(data2_iq)
    index_sj, index_iq = np.array(index_sj), np.array(index_iq)
    week_sj, week_iq = np.reshape(week_sj, (-1, 1)), np.reshape(week_iq, (-1, 1))

    return (data2_sj, data2_iq), (index_sj, index_iq), (week_sj, week_iq)


def read_labels(filename):

    label_sj = []
    label_iq = []

    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            city, year, weekofyear, totalcases = line.split(',')
            if city == 'sj':
                label_sj.append(int(totalcases))
            elif city == 'iq':
                label_iq.append(int(totalcases))
            else:
                print('unknown city')

    label_sj, label_iq = np.array(label_sj), np.array(label_iq)

    return label_sj, label_iq
