#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:53:07 2018

@author: edip.demirbilek
"""
import pandas as pd
import glob

df3 = pd.DataFrame()

subjects_dir = "/Users/edip.demirbilek/PrivateProjects/MultimediaRegressionModels/dataset/subject_details/"
mos_dir = "/Users/edip.demirbilek/PrivateProjects/MultimediaRegressionModels/dataset/"
mos_bitstream_file = "bitstream_dataset.csv"
mos_base_file = "base_dataset.csv"
mos_bitstream_subjects_file = "bitstream_subjects_dataset.csv"

all_files = glob.glob(subjects_dir + "/*.csv")

df_from_each_file = (pd.read_csv(f) for f in all_files)
# DataFrame (4564, 2)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

# DataFrame (160, 7)
df = pd.read_csv(mos_dir+mos_base_file)
# DataFrame (160, 126)
df2 = pd.read_csv(mos_dir+mos_bitstream_file)
df3 = pd.DataFrame()

# concatenated_df -> DataFrame (4564, 2)
for index, row in concatenated_df.iterrows():
    # get FPS, Q, NR, PLR falues from the GeneralFileName column
    # get MOS from MOS column
    tokens = row['GeneralFileName'].split('_')
    FPS = tokens[5].split('fps')[1]
    Q = tokens[6].split('q')[1]
    NR = tokens[7].split('nr')[1]
    PLR = tokens[8].split('plr')[1]
    MOS = row['MOS']

    # find actual video and audio packet loss rates
    # row_base -> DataFrame (1, 7)
    row_base = df.loc[(df['FPS'] == float(FPS)) &
                      (df['QP'] == float(Q)) &
                      (df['NR'] == float(NR)) &
                      (df['PLR'] == float(PLR))]

    vPLR = row_base['vPLR']
    aPLR = row_base['aPLR']

    # find bitstream row for selected FPS, Q, NR, vPLR, aPLR
    # row bitstream -> DataFrame (1, 126)
    row_bitstream = df2.loc[(df2['VideoFrameRate'] == float(FPS)) &
                            (df2['QP'] == float(Q)) &
                            (df2['NR'] == float(NR)) &
                            (df2['VideoPacketLossRate'] == float(vPLR)) &
                            (df2['AudioPacketLossRate'] == float(aPLR))]

    # take entire bitstream row except MOS
    # entire_row -> DataFrame (1, 125)
    entire_row = row_bitstream.loc[:, :'DTS30aFrameCountDiff']
    entire_row = entire_row.reset_index(drop=True)

    # concat bitstream row above with the MOS from user
    # MOS_column -> DataFrame (1, 1)
    MOS_column = pd.DataFrame(columns=['MOS'])
    MOS_column.loc[0] = [MOS]
    # final_row -> DataFrame (1, 126)
    final_row = pd.concat([entire_row, MOS_column], axis=1)

    # append bitstream row + MOS to global result
    df3 = df3.append(final_row)

# df3 -> DataFrame (4564, 126)
df3.to_csv(mos_dir+mos_bitstream_subjects_file, index=False)
