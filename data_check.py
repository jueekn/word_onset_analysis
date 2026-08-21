"""data_check.py — pre-flight per-session data validation.

For each session enumerated by cmlreaders, checks that bipolar pair tables,
EEG, and event metadata are all loadable and well-formed; writes per-session
data_check JSON + electrode_information sidecars. Aggregator
build_sess_list_df_data_check() flips include=False on any failure.

Dominated by cmlreaders / ptsa / pandas / mne method chains; narrow the
four library-stub-noise rules at file scope.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportUnknownLambdaType=warning, reportUnknownParameterType=warning, reportMissingParameterType=warning, reportOptionalMemberAccess=warning, reportOptionalSubscript=warning, reportOptionalIterable=warning, reportIndexIssue=warning, reportCallIssue=warning, reportPossiblyUnboundVariable=warning, reportUnusedVariable=warning, reportArgumentType=warning, reportAssignmentType=warning, reportReturnType=warning, reportGeneralTypeIssues=warning
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import os
import h5py  # pyright: ignore[reportMissingTypeStubs]

import cmlreaders as cml  # pyright: ignore[reportMissingTypeStubs]

from os.path import join, exists as ex

import pandas as pd

from ptsa.data.readers import BaseEventReader, EEGReader, TalReader  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.filters import MonopolarToBipolarMapper  # pyright: ignore[reportMissingTypeStubs]

from cstat import *  # noqa: F401,F403
from misc import *  # noqa: F401,F403
from misc import ftag  # explicit for type checker
from matrix_operations import *  # noqa: F401,F403

from helper import get_phase, timebin_phase_timeseries, regionalize_electrodes_by_type
from project_paths import RESAMPLE_HZ, PAIR_DISTANCE_THRESHOLD_MM

NDArrayAny = npt.NDArray[Any]

# data_check-internal QC constants. Non-critical (validation mechanics / cohort
# scope, not scientific tunables), so they live here rather than config.yaml —
# but are module-level + importable for other modules and tests.
DATA_CHECK_EEG_WINDOW_MS: tuple[int, int] = (-1500, 1500)
DATA_CHECK_PHASE_FREQ_HZ: float = 3.0
EEGOFFSET_MONOTONICITY_TOL_SAMPLES: int = 10
MSTIME_EEGOFFSET_RATIO_TOL: float = 0.01
COHORT_EXPERIMENTS: tuple[str, ...] = ("FR1", "catFR1", "pyFR")


def build_sess_list_df_initial(root_dir: str) -> pd.DataFrame:
    """
    Creates sess_list_df_initial.csv in root_dir 
    """
    ethan_subject_list = [['R1031M', 'FR1'],['R1033D', 'FR1'],['R1042M', 'FR1'],['R1044J', 'FR1'],['R1039M', 'FR1'],['R1048E', 'FR1'],['R1035M', 'FR1'],['R1030J', 'FR1'],['R1050M', 'FR1'],['R1053M', 'FR1'],['R1036M', 'FR1'],['R1018P', 'FR1'],['R1020J', 'FR1'],['R1026D', 'FR1'],['R1032D', 'FR1'],['R1056M', 'FR1'],['R1002P', 'FR1'],['R1027J', 'FR1'],['R1045E', 'FR1'],['R1057E', 'FR1'],['R1063C', 'FR1'],['R1001P', 'FR1'],['R1003P', 'FR1'],['R1015J', 'FR1'],['R1049J', 'FR1'],['R1022J', 'FR1'],['R1074M', 'FR1'],['R1069M', 'FR1'],['R1076D', 'FR1'],['R1102P', 'FR1'],['R1023J', 'FR1'],['R1155D', 'FR1'],['R1162N', 'FR1'],['R1086M', 'FR1'],['R1149N', 'FR1'],['R1034D', 'FR1'],['R1051J', 'FR1'],['R1159P', 'FR1'],['R1101T', 'FR1'],['R1178P', 'FR1'],['R1104D', 'FR1'],['R1089P', 'FR1'],['R1105E', 'FR1'],['R1060M', 'FR1'],['R1186P', 'FR1'],['R1112M', 'FR1'],['R1202M', 'FR1'],['R1066P', 'FR1'],['R1084T', 'FR1'],['R1096E', 'FR1'],['R1176M', 'FR1'],['R1187P', 'FR1'],['R1169P', 'FR1'],['R1081J', 'FR1'],['R1006P', 'FR1'],['R1164E', 'FR1'],['R1196N', 'FR1'],['R1127P', 'FR1'],['R1010J', 'FR1'],['R1175N', 'FR1'],['R1148P', 'FR1'],['R1203T', 'FR1'],['R1193T', 'FR1'],['R1167M', 'FR1'],['R1075J', 'FR1'],['R1177M', 'FR1'],['R1080E', 'FR1'],['R1161E', 'FR1'],['R1068J', 'FR1'],['R1054J', 'FR1'],['R1195E', 'FR1'],['R1189M', 'FR1'],['R1131M', 'FR1'],['R1200T', 'FR1'],['R1100D', 'FR1'],['R1142N', 'FR1'],['R1114C', 'FR1'],['R1247P', 'FR1'],['R1128E', 'FR1'],['R1138T', 'FR1'],['R1059J', 'FR1'],['R1067P', 'FR1'],['R1215M', 'FR1'],['R1111M', 'FR1'],['R1171M', 'FR1'],['R1106M', 'FR1'],['R1129D', 'FR1'],['R1146E', 'FR1'],['R1098D', 'FR1'],['R1130M', 'FR1'],['R1115T', 'FR1'],['R1150J', 'FR1'],['R1173J', 'FR1'],['R1154D', 'FR1'],['R1136N', 'FR1'],['R1137E', 'FR1'],['R1204T', 'FR1'],['R1174T', 'FR1'],['R1230J', 'FR1'],['R1122E', 'FR1'],['R1229M', 'FR1'],['R1158T', 'FR1'],['R1163T', 'FR1'],['R1250N', 'FR1'],['R1147P', 'FR1'],['R1166D', 'FR1'],['R1241J', 'FR1'],['R1123C', 'FR1'],['R1120E', 'FR1'],['R1083J', 'FR1'],['R1113T', 'FR1'],['R1226D', 'FR1'],['R1228M', 'FR1'],['R1062J', 'FR1'],['R1232N', 'FR1'],['R1217T', 'FR1'],['R1151E', 'FR1'],['R1070T', 'FR1'],['R1121M', 'FR1'],['R1222M', 'FR1'],['R1168T', 'FR1'],['R1172E', 'FR1'],['R1223E', 'FR1'],['R1134T', 'FR1'],['R1077T', 'FR1'],['R1094T', 'FR1'],['R1260D', 'FR1'],['R1061T', 'FR1'],['R1243T', 'FR1'],['R1191J', 'FR1'],['R1240T', 'FR1'],['R1231M', 'FR1'],['R1234D', 'FR1'],['R1135E', 'FR1'],['R1004D', 'catFR1'],['R1065J', 'FR1'],['R1236J', 'FR1'],['R1016M', 'catFR1'],['R1021D', 'catFR1'],['R1028M', 'catFR1'],['R1041M', 'catFR1'],['R1029W', 'catFR1'],['R1119P', 'catFR1'],['R1107J', 'catFR1'],['R1141T', 'catFR1'],['R1144E', 'catFR1'],['R1188C', 'catFR1'],['R1180C', 'catFR1'],['R1157C', 'catFR1'],['R1181E', 'catFR1'],['R1190P', 'catFR1'],['R1227T', 'catFR1'],['R1192C', 'catFR1'],['R1221P', 'catFR1'],['R1239E', 'catFR1'],['R1273D', 'catFR1'],['TJ017', 'pyFR'],['TJ018_2', 'pyFR'],['TJ018', 'pyFR'],['TJ019', 'pyFR'],['TJ021', 'pyFR'],['TJ020', 'pyFR'],['TJ024', 'pyFR'],['TJ027', 'pyFR'],['TJ022', 'pyFR'],['TJ028', 'pyFR'],['TJ029', 'pyFR'],['TJ033', 'pyFR'],['TJ030', 'pyFR'],['TJ031', 'pyFR'],['TJ025', 'pyFR'],['TJ032', 'pyFR'],['TJ036', 'pyFR'],['TJ034', 'pyFR'],['TJ037', 'pyFR'],['TJ035_1', 'pyFR'],['TJ038_1', 'pyFR'],['TJ038', 'pyFR'],['TJ040_1', 'pyFR'],['TJ040', 'pyFR'],['TJ042', 'pyFR'],['TJ041', 'pyFR'],['TJ043', 'pyFR'],['TJ041_2', 'pyFR'],['TJ039', 'pyFR'],['TJ044', 'pyFR'],['TJ045', 'pyFR'],['TJ047', 'pyFR'],['TJ048', 'pyFR'],['TJ049', 'pyFR'],['TJ052', 'pyFR'],['TJ053_2', 'pyFR'],['TJ053_3', 'pyFR'],['TJ055', 'pyFR'],['TJ054', 'pyFR'],['TJ056', 'pyFR'],['TJ058', 'pyFR'],['TJ057', 'pyFR'],['TJ059', 'pyFR'],['TJ061', 'pyFR'],['TJ064', 'pyFR'],['TJ062_2', 'pyFR'],['TJ060', 'pyFR'],['TJ065', 'pyFR'],['TJ064_1', 'pyFR'],['TJ066', 'pyFR'],['TJ068', 'pyFR'],['TJ069_1', 'pyFR'],['TJ069', 'pyFR'],['TJ071', 'pyFR'],['TJ072', 'pyFR'],['TJ074_1', 'pyFR'],['TJ073', 'pyFR'],['TJ074', 'pyFR'],['TJ077', 'pyFR'],['TJ078_1', 'pyFR'],['TJ078', 'pyFR'],['TJ081', 'pyFR'],['TJ079', 'pyFR'],['TJ080', 'pyFR'],['UP016', 'pyFR'],['UP017', 'pyFR'],['UP019', 'pyFR'],['UP020', 'pyFR'],['UP022', 'pyFR'],['UP021', 'pyFR'],['UP028', 'pyFR'],['UP024', 'pyFR'],['UP029', 'pyFR'],['UP034', 'pyFR'],['UP036', 'pyFR'],['UP038', 'pyFR'],['UP037', 'pyFR'],['UP042', 'pyFR'],['UP041', 'pyFR'],['UP040', 'pyFR'],['UP044_1', 'pyFR'],['UP043', 'pyFR'],['UP044', 'pyFR'],['UP045', 'pyFR'],['R1153T', 'FR1'],['R1156D', 'FR1'],['R1207J', 'FR1'],['TJ051', 'pyFR'],['R1052E_1', 'FR1'],['R1052E_2', 'FR1'],['R1055J', 'FR1'],['R1059J_1', 'FR1'],['R1063C_1', 'FR1'],['R1066P_1', 'FR1'],['R1083J_1', 'FR1'],['R1092J_2', 'FR1'],['R1108J_1', 'FR1'],['R1118N_1', 'FR1'],['R1124J_1', 'FR1'],['R1127P_1', 'FR1'],['R1145J_1', 'FR1'],['TJ086', 'FR1'],['UT009', 'FR1'],['R1170J_1', 'FR1'],['R1171M_1', 'FR1'],['R1092J_3', 'FR1'],['R1185N_1', 'FR1'],['R1201P_1', 'FR1'],['R1216E_1', 'FR1'],['R1291M', 'FR1'],['R1093J', 'catFR1'],['R1013E_1', 'catFR1'],['R1024E_1', 'catFR1'],['R1092J', 'catFR1'],['R1108J', 'catFR1'],['R1127P_2', 'catFR1'],['R1135E_1', 'catFR1'],['R1138T_1', 'catFR1'],['TJ084', 'catFR1'],['R1191J_1', 'catFR1'],['R1247P_1', 'catFR1'],['R1269E_1', 'catFR1'],['R1278E_1', 'catFR1'],['FR150', 'pyFR'],['FR160', 'pyFR'],['FR190', 'pyFR'],['FR220', 'pyFR'],['FR230', 'pyFR']]
    ethan_subject_list = np.asarray(ethan_subject_list)
    ethan_sess_list = []
    data_index = cml.get_data_index()
    for subexp in ethan_subject_list:
        sub, exp = subexp
        sess_list = data_index.query('subject == @sub & experiment == @exp')[['session', 'localization', 'montage']].to_numpy()
        for r in sess_list:
            sess, loc, mon = r
            sess, loc, mon = int(sess), int(loc), int(mon)
            row = np.asarray([sub, exp, sess, loc, mon], dtype='object')
            ethan_sess_list.append(row)

    ethan_sess_list = np.asarray(ethan_sess_list)
    ethan_sess_list = pd.DataFrame(
        ethan_sess_list,
        columns=['subject', 'experiment', 'session', 'localization', 'montage']
    )

    data_index = cml.get_data_index()
    cmlreaders_sess_list = data_index.query(f'experiment in {list(COHORT_EXPERIMENTS)}')[['subject', 'experiment', 'session', 'localization', 'montage']]

    sess_list_df = pd.concat([cmlreaders_sess_list, ethan_sess_list], axis=0).drop_duplicates(ignore_index=True)
    sess_list_df = sess_list_df.rename({'subject': 'sub', 'experiment': 'exp', 'session': 'sess', 'localization': 'loc', 'montage': 'mon'}, axis=1)

    sess_list_df.to_csv(join(root_dir, 'sess_list_df_initial.csv'), index=False)

    return sess_list_df

def channels_to_drop(root_dir: str) -> pd.DataFrame | pd.Series:
    dfrows=[('R1063C', 'FR1', 1, 0, 1),
    ('R1216E', 'FR1', 1, 0, 1),
    ('R1216E', 'FR1', 2, 0, 1),
    ('R1229M', 'catFR1', 0, 0, 0),
    ('R1354E', 'FR1', 0, 0, 0),
    ('R1354E', 'FR1', 1, 0, 0),
    ('R1354E', 'catFR1', 0, 0, 0),
    ('R1354E', 'catFR1', 1, 0, 0),
    ('R1354E', 'catFR1', 2, 0, 0),
    ('R1354E', 'catFR1', 3, 0, 0),
    ('R1367D', 'FR1', 0, 0, 0),
    ('R1367D', 'FR1', 1, 0, 0),
    ('R1367D', 'catFR1', 0, 0, 0),
    ('R1394E', 'FR1', 0, 0, 0),
    ('R1394E', 'FR1', 1, 1, 1),
    ('R1394E', 'catFR1', 1, 1, 1),
    ('R1394E', 'catFR1', 2, 1, 1),
    ('R1478T', 'catFR1', 0, 0, 0),
    ('R1626S', 'catFR1', 0, 0, 0),
    ('R1626S', 'catFR1', 2, 0, 0),
    ('R1626S', 'catFR1', 3, 0, 0),
    ('R1626S', 'catFR1', 4, 0, 0),
    ('R1626S', 'catFR1', 6, 0, 0),
    ('R1626S', 'catFR1', 7, 0, 0),
    ('R1626S', 'catFR1', 8, 0, 0)]
    channels_to_drop_df = pd.DataFrame([], columns=['sub', 'exp', 'sess', 'loc', 'mon', 'channels'])
    for i, dfrow in enumerate(dfrows):
        channels = []
        reader = cml.CMLReader(*dfrow)
        pairs = reader.load('pairs')
        events = reader.load('events').query('type=="WORD"').iloc[0:1]
        eeg = reader.load_eeg(events, 0, 1000, scheme=pairs)
        for ch in pairs.label.values:
            if ch not in eeg.channels:
                channels.append(ch)
        channels_to_drop_df.loc[i, ['sub', 'exp', 'sess', 'loc', 'mon']] = dfrow
        channels_to_drop_df.loc[i, 'channels'] = channels
    channels_to_drop_df.to_json(join(root_dir, 'channels_to_drop_df.json'))
    return channels_to_drop_df

def check_pairs(dfrow: pd.Series, root_dir: str) -> tuple[Any, ...]:
    def get_mat_pairs(sub, mon_):

        tal_path=f'/data/eeg/{sub}{mon_}/tal/{sub}{mon_}_talLocs_database_bipol.mat'
        df = load_mat(tal_path)
        k = 'bpTalStruct' if 'bpTalStruct' in df.keys() else 'subjTalEvents'
        df = pd.DataFrame(df[k][0])
        df.rename({'bpDistance': 'distance'}, axis=1, inplace=True)
        channel_info=np.asarray([[df['channel'][i][0][0],df['channel'][i][0][1],df['tagName'][i][0],str(df['Loc1'][i][0]),str(df['Loc2'][i][0]),str(df['Loc3'][i][0]),str(df['Loc4'][i][0]),str(df['Loc5'][i][0]),df['x'][i][0][0],df['y'][i][0][0],df['z'][i][0][0],df['distance'][i][0][0]] for i in range(len(df))])
        pairs=pd.DataFrame(channel_info, columns = ['contact_1','contact_2','label','Loc1','Loc2','Loc3','Loc4','Loc5','x','y','z','distance'])
        pairs[['x', 'y', 'z', 'distance']] = pairs[['x', 'y', 'z', 'distance']].astype(float)

        for reg_col, surf_col in zip(['mat.avg.region', 'mat.ind.region'], ['avgSurf', 'indivSurf']):
            if surf_col in df.columns: 
                reg_col_values = [str(df[surf_col][i]['anatRegion'][0][0][0]) if len(df[surf_col][i]['anatRegion'][0][0]) > 0 else 'nan' for i in range(len(df))]
                if not np.all(np.isin(np.char.lower(reg_col_values), ['nan', 'none', 'unknown'])):
                    pairs[reg_col] = reg_col_values

        assert np.any(np.isin(['mat.avg.region', 'mat.ind.region'], pairs.columns)), f'{surf_col} information not available'

        for reg_col, surf_col in zip(['mat.avg.region', 'mat.ind.region'], ['avgSurf', 'indivSurf']):
            if reg_col not in df.columns: continue
            for reg_corr_col_, surf_corr_col, in zip(['.corrected.region', '.snap.region', '.dural.region'], ['anatRegion_Dykstra', 'anatRegion_dural', 'anatRegion_snap']):
                if surf_corr_col not in df[surf_col][0].dtype.names: continue  # pyright: ignore[reportOperatorIssue]
                reg_corr_col = reg_corr_col_.replace('.region', col)
                pairs[reg_corr_col] = [str(df[surf_col][i][surf_corr_col][0][0][0]) if len(df[surf_col][i][surf_corr_col][0][0]) > 0 else 'nan' for i in range(len(df))]

        def get_hemisphere(pair):

            if 'left' in pair['Loc2'].lower(): return 'L'
            if 'right' in pair['Loc2'].lower(): return 'R'

            if str(pair['x']) == 'nan': return 'nan'
            elif pair['x'] < 0: return 'L'
            elif pair['x'] > 0: return 'R'

            return 'nan'

        pairs['hemisphere'] = pairs.apply(lambda x: get_hemisphere(x), axis=1)
        pairs[['contact_1', 'contact_2']] = pairs[['contact_1', 'contact_2']].astype(int)

        def get_tal_region(r):
            
            Loc3_to_Loc5_dict = {'Parahippocampal Gyrus': ['Hippocampus', 'Amygdala'],
                                 'Uncus': ['Amygdala'],
                                 'Lentiform Nucleus': ['Lateral Globus Pallidus', 'Putamen', 'Medial Globus Pallidus']}

            return (r['Loc3'] + ' ' + r['Loc5']) if r['Loc5'] in Loc3_to_Loc5_dict.get(r['Loc3'], []) else r['Loc3']

        pairs['mat.tal.region'] = pairs.apply(lambda r: get_tal_region(r), axis=1)

        return pairs
    
    try:
        sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
        mon_ = '' if mon==0 else f'_{mon}'
        reader = cml.CMLReader(subject=sub, 
                               experiment=exp, 
                               session=sess,
                               localization=loc,
                               montage=mon)
        atlas = np.nan
        contacts_source = np.nan
        try: 
            pairs = reader.load('pairs')

            if 'type_1' in pairs.columns: pairs = pairs[~(pairs['type_1'] == 'uD')].reset_index(drop=True) #throw out any microwire channels
            
            pairs.rename({'bpDistance': 'distance'}, axis=1, inplace=True)
            
            if 'distance' not in pairs.columns:
                
                def get_contact_label(label):
                    split_label = label.split('-')
                    if len(split_label) == 2: return split_label
                    split_idx = int(np.median(np.where(np.asarray([*label]) == '-')))
                    return [label[:split_idx], label[(split_idx+1):]]
                
                atlas = 'avg' if (np.all(np.isin(['avg.x', 'avg.y', 'avg.z'], pairs.columns))) and (pd.isna(pairs[['avg.x', 'avg.y', 'avg.z']]).values.sum() == 0) else 'tal'
                atlas_coordinates = ['avg.x', 'avg.y', 'avg.z']
                pairs[['contact_label_1', 'contact_label_2']] = pairs.apply(lambda pair: get_contact_label(pair['label']), result_type='expand', axis=1)
                
                def load_localization_contacts(reader):
                    
                    contacts = reader.load('localization').loc['contacts']
                    for iCol, col in enumerate(['avg.x', 'avg.y', 'avg.z']): contacts[col] = [x[iCol] for x in contacts['coordinate_spaces.fsaverage.raw'].values]
                    return contacts
                
                if tuple(dfrow) in [('R1373T', 'FR1', 0, 0, 0), 
                                    ('R1478T', 'catFR1', 0, 0, 0), 
                                    ('R1490T', 'catFR1', 0, 0, 0)]: #these sessions don't have contacts.json available
                    contacts = load_localization_contacts(reader)
                    contacts_source = 'localization'
                else:
                    contacts = reader.load('contacts').set_index('label')
                    contacts_source = 'contacts'
#                     if len(contacts.index) < len(np.unique(pairs[['contact_label_1', 'contact_label_2']].values.ravel())):
                    if np.any(~np.isin(np.unique(pairs[['contact_label_1', 'contact_label_2']].values.ravel()), contacts.index.values)):
                        contacts = load_localization_contacts(reader)
                        contacts_source = 'localization'

                pairs['distance'] = pairs.apply(lambda pair: np.sqrt(np.sum((contacts.loc[pair[f'contact_label_2'], atlas_coordinates].values.astype(float) - contacts.loc[pair[f'contact_label_1'], atlas_coordinates].values.astype(float))**2)), axis=1)
            if ex(f'/data/eeg/{sub}{mon_}/tal/{sub}{mon_}_talLocs_database_bipol.mat'):
                mat_pairs = get_mat_pairs(sub, mon_)
                mat_pairs_available = True
                mat_pairs.set_index('label', inplace=True)
                pairs['tal.region'] = mat_pairs['mat.tal.region']
                for col in ['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5', 'mat.ind.corrected.region', 'mat.ind.snap.region', 'mat.ind.dural.region', 'mat.ind.region', 'mat.avg.corrected.region', 'mat.avg.snap.region', 'mat.avg.dural.region', 'mat.avg.region', 'mat.tal.region']:
                    if (col not in pairs.columns) and (col in mat_pairs.columns): 
                        pairs[col] = np.nan
                        for i in range(len(pairs)):
                            label = pairs.loc[i, 'label']
                            if label in mat_pairs.index.values: 
                                pairs.loc[i, col] = mat_pairs.loc[label, col]
            else:
                mat_pairs_available = False
            pairs_data_source = 'cmlreaders'
            try: 
                localization = reader.load('localization')
                assert np.any(np.isin(['atlases.dk', 'atlases.dkavg', 'atlases.mtl', 'atlases.whole_brain'], localization))
            except: 
                localization = []
                assert np.any(np.isin(['stein.region', 'das.region', 'wb.region', 'dk.region', 'ind.corrected.region', 'ind.snap.region', 'ind.dural.region', 'ind.region', 'avg.corrected.region', 'avg.snap.region', 'avg.dural.region', 'avg.region', 'mni.region'], pairs.columns))
        except Exception as e:
            print(e)
            pairs = get_mat_pairs(sub, mon_)
            mat_pairs_available = True
            pairs_data_source = 'mat'
            localization = []
    
        distance_threshold = PAIR_DISTANCE_THRESHOLD_MM
        assert pd.isna(pairs['distance']).sum() == 0, 'missing distance values for pairs'
        long_distance_pairs_count = len(pairs.query('distance > @distance_threshold'))
        pairs = pairs.query('distance <= @distance_threshold').reset_index(drop=True)
        
        channels_to_drop_df = pd.read_json(join(root_dir, 'channels_to_drop_df.json')).set_index(['sub', 'exp', 'sess', 'loc', 'mon'])
        if tuple(dfrow) in channels_to_drop_df.index:
            channels_to_drop = channels_to_drop_df.loc[tuple(dfrow), 'channels']
            pairs = pairs[~np.isin(pairs['label'], channels_to_drop)].reset_index(drop=True)

        return pairs, localization, pairs_data_source, mat_pairs_available, long_distance_pairs_count, atlas, contacts_source, True
    except Exception as e:
        print(e)
        return None, None, np.nan, np.nan, np.nan, np.nan, np.nan, False
    
def check_regionalizations(
    pairs: pd.DataFrame, localization: pd.DataFrame | None
) -> tuple[Any, bool]:
    
    try:
        regionalizations = regionalize_electrodes_by_type(pairs, localization)
        return regionalizations, True
    except:
        return None, False
    
def check_eeg(
    dfrow: pd.Series, pairs: pd.DataFrame,
    start: int = DATA_CHECK_EEG_WINDOW_MS[0], end: int = DATA_CHECK_EEG_WINDOW_MS[1]
) -> tuple[Any, bool, Any]:
    
    
    data_check = pd.Series({})
    global errors
    errors = []
    
    def check_eegoffset_monotonicity(events):
        
        return np.all(events.groupby('eegfile')['eegoffset'].diff().fillna(0) >= -EEGOFFSET_MONOTONICITY_TOL_SAMPLES)
    
    def check_mstime_eegoffset_match(events):

        events = events.query('type in ["WORD", "REC_WORD", "REC_WORD_VV"]')[['eegfile', 'mstime', 'rectime', 'eegoffset']].reset_index()
        events['mstime_diff'] = events['mstime'].diff()/1000
        events['eegoffset_time_diff'] = events['eegoffset'].diff()/sr
        eegfiles = events.eegfile.values
        eegfile_change_idxs = np.asarray([i for i, x in enumerate(eegfiles) if eegfiles[i] != eegfiles[i-1]])
        if len(eegfile_change_idxs) > 0:                                  
            events.loc[eegfile_change_idxs, ['mstime_diff', 'eegoffset_time_diff']] = np.nan

        mean_mstime_eegoffset_ratio = (events['mstime_diff']/events['eegoffset_time_diff']).mean()
        median_mstime_eegoffset_ratio = (events['mstime_diff']/events['eegoffset_time_diff']).median()
        lo, hi = 1.0 - MSTIME_EEGOFFSET_RATIO_TOL, 1.0 + MSTIME_EEGOFFSET_RATIO_TOL
        if (lo <= mean_mstime_eegoffset_ratio <= hi) and (lo <= median_mstime_eegoffset_ratio <= hi):
            mstime_eegoffset_match = True
        else:
            mstime_eegoffset_match = False
        return mean_mstime_eegoffset_ratio, median_mstime_eegoffset_ratio, mstime_eegoffset_match
    
    try:
        #Load EEG
        try: 
            sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
            reader = cml.CMLReader(subject=sub, 
                                   experiment=exp, 
                                   session=sess,
                                   localization=loc,
                                   montage=mon)
            events = reader.load('events')
            empty_eegfile = np.any(np.isin(['', '[]'], np.unique(events.query('type in ["WORD", "REC_START", "REC_WORD", "REC_WORD_VV"]')['eegfile'])))
            try: 
                assert not empty_eegfile, 'cmlreaders events have empty eegfile values'
            except AssertionError as e:
                errors.append(e)
                assert False
            
            en_events = events.query('type == "WORD" & eegfile != ""').iloc[0:1]
            eeg = reader.load_eeg(en_events, start, end, scheme=pairs)
            try: 
                assert len(eeg.channels) == len(pairs), 'Number of channels in EEG does not match number of channels in pairs'
            except AssertionError as e:
                errors.append(e)
                assert False
            
            sr = float(eeg.samplerate)
            evs_data_source = eeg_data_source = 'cmlreaders'
            eeg = eeg.to_ptsa()
            
            events.sort_values(by='mstime', inplace=True)
            eegoffset_monotonic = check_eegoffset_monotonicity(events.query('type in ["WORD", "REC_START", "REC_WORD", "REC_WORD_VV"]'))
            try: 
                assert eegoffset_monotonic, 'cmlreaders events have nonmonotonic eegoffset values'
            except AssertionError as e:
                errors.append(e)
                assert False
            
            mean_mstime_eegoffset_ratio, median_mstime_eegoffset_ratio, mstime_eegoffset_match = check_mstime_eegoffset_match(events)
            
            try:
                assert mstime_eegoffset_match, 'cmlreaders have mstime/eegoffset mismatch'
            except AssertionError as e:
                errors.append(e)
                assert False
            
        except: 
            
            evs_data_source, eeg_data_source, sr, eeg, events = check_ptsa_eeg(dfrow, pairs, start, end)
            events = pd.DataFrame.from_records(events)
            events.sort_values(by='mstime', inplace=True)
            eegoffset_monotonic = check_eegoffset_monotonicity(events.query('type in ["WORD", "REC_START", "REC_WORD", "REC_WORD_VV"]'))
            try: 
                assert eegoffset_monotonic, 'cmlreaders events have nonmonotonic eegoffset values'
            except AssertionError as e:
                errors.append(e)
                assert False
            
            mean_mstime_eegoffset_ratio, median_mstime_eegoffset_ratio, mstime_eegoffset_match = check_mstime_eegoffset_match(events)
            
            try:
                assert mstime_eegoffset_match, 'cmlreaders have mstime/eegoffset mismatch'
            except AssertionError as e:
                errors.append(e)
                assert False
        
        eeg = eeg.resampled(RESAMPLE_HZ)  # match production (config: resample_hz)
        
        dim_map = {}
        if 'events' in eeg.dims: dim_map['events'] = 'event'
        if 'channels' in eeg.dims: dim_map['channels'] = 'channel'
        eeg = eeg.rename(dim_map)
        eeg = eeg.transpose('event', 'channel', 'time')
        
        eeg_channels = eeg.channel.values

        data_check['eeg_error'] = np.nan
        for var in ['evs_data_source', 'eeg_data_source', 'sr', 'eeg_channels']:
            data_check[var] = locals()[var]

        return eeg, True, data_check
    
    except Exception as e:
        errors.append(e)
        data_check['eeg_error'] = globals()['errors']
        for var in ['evs_data_source', 'eeg_data_source', 'sr', 'eeg_channels']:
            data_check[var] = np.nan
        return None, False, data_check
    
def check_ptsa_eeg(
    dfrow: pd.Series, pairs: pd.DataFrame, start: int, end: int
) -> Any:
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    exp_dict = {'FR1': 'RAM_FR1', 'catFR1': 'RAM_CatFR1', 'pyFR': 'pyFR'}
    exp = exp_dict[exp] #for tal_reader path name
    mon_ = '' if mon==0 else f'_{mon}' #for tal_reader path name
        
    def load_ptsa_eeg(events):        
        
        en_events = events[(events.session==sess) & (events.type=='WORD') & (events.eegfile!='')][0:1]

        tal_reader = TalReader(filename=f'/data/eeg/{sub}{mon_}/tal/{sub}{mon_}_talLocs_database_bipol.mat')
        channels = tal_reader.get_monopolar_channels()
        eeg = EEGReader(events=en_events, channels=channels,
                    start_time=start/1000, end_time=end/1000).read()
 
        bipolar_pairs = tal_reader.get_bipolar_pairs()
        pair_tuples_select = [tuple((int(pair[0]), int(pair[1]))) for pair in pairs[['contact_1', 'contact_2']].values]
        bipolar_pairs = np.asarray([pair for pair in bipolar_pairs if tuple((int(pair[0]), int(pair[1]))) in pair_tuples_select], dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
        mapper= MonopolarToBipolarMapper(bipolar_pairs=bipolar_pairs)
        eeg = mapper.filter(timeseries=eeg)
        eeg = eeg.transpose('events', 'channels', 'time')
        
        return eeg
    
    try:
        events = BaseEventReader(filename=f'/data/events/{exp}/{sub}{mon_}_events.mat', use_reref_eeg=False).read()
        empty_eegfile = np.any(np.isin(['', '[]'], np.unique(events[np.isin(events.type, ['WORD', 'REC_START', 'REC_WORD', 'REC_WORD_VV'])]['eegfile'])))
        try: 
            assert not empty_eegfile, 'ptsa events have empty eegfile values'
        except AssertionError as e:
            globals()['errors'].append(e)
        
        eeg = load_ptsa_eeg(events)
        evs_data_source = 'ptsa'
        eeg_data_source = 'ptsa'
    except Exception as e:
        print(e)
        globals()['errors'].append(e)
        reader = cml.CMLReader(subject=sub, 
                       experiment=exp, 
                       session=sess,
                       localization=loc,
                       montage=mon)
        events = reader.load('events')
        events = events.to_records()
        empty_eegfile = '' in np.unique(events[np.isin(events.type, ['WORD', 'REC_START', 'REC_WORD', 'REC_WORD_VV'])]['eegfile'])
        try: 
            assert not empty_eegfile, 'cmlreaders events have empty eegfile values'
        except AssertionError as e:
            globals()['errors'].append(e)
        
        eeg = load_ptsa_eeg(events)
        evs_data_source = 'cmlreaders'
        eeg_data_source = 'ptsa'
    
    sr = float(eeg.samplerate)
    events = events[np.isin(events.type, ['WORD', 'REC_WORD', 'REC_WORD_VV'])]

    return evs_data_source, eeg_data_source, sr, eeg, events
        
def check_phase(eeg: Any) -> dict[str, Any]:
    
    try:
        freqs = [DATA_CHECK_PHASE_FREQ_HZ]
        phase = get_phase(eeg, freqs)
        sr = float(phase.samplerate)
        timebinned_phase = timebin_phase_timeseries(phase.data, sr)
        return phase, True
    except:
        return None, False
    
def _trace_is_constant(trace: Any, chunk: int = 65536) -> bool:
    """True iff every sample in ``trace`` equals the first (i.e. ``var == 0``).

    Bit-equivalent to ``np.var(trace) == 0`` for the constant-channel test, but
    scans in chunks and returns as soon as a differing sample is seen — so a
    live channel (the overwhelming majority) reads only its first chunk instead
    of the whole recording. Empty traces return False to match the legacy
    ``np.var([]) == 0`` (nan) -> kept behavior.
    """
    n = len(trace)
    if n == 0:
        return False
    first = trace[0]
    for i in range(0, n, chunk):
        if (np.asarray(trace[i:i + chunk]) != first).any():
            return False
    return True


def remove_constant_channels(
    dfrow: pd.Series,
    pairs: pd.DataFrame,
    evs_data_source: str,
    eeg_data_source: str,
) -> tuple[Any, bool]:

    sub, exp, sess, loc, mon = dfrow
    mon_ = '' if mon==0 else f'_{mon}'

    if evs_data_source == 'cmlreaders':
        reader = cml.CMLReader(*dfrow)
        eegfile = reader.load('events').query('type=="WORD"').iloc[0].eegfile
    elif evs_data_source == 'ptsa':
        exp_dict = {'FR1': 'RAM_FR1', 'catFR1': 'RAM_CatFR1', 'pyFR': 'pyFR'}
        exp = exp_dict[exp] #for tal_reader path name
        events = BaseEventReader(filename=f'/data/events/{exp}/{sub}{mon_}_events.mat', use_reref_eeg=False).read()
        eegfile = events[(events.session==sess) & (events.type=='WORD')][0].eegfile

    if eeg_data_source == 'cmlreaders': # If the EEG data source is cmlreaders, we are going to load up the EEG file from /protocols/r1/subjects
        raw_eeg_dir = os.path.join('/protocols/r1/subjects', sub, 'experiments', exp, 'sessions', str(sess), 'ephys/current_processed/noreref')
    elif eeg_data_source == 'ptsa': 
        raw_eeg_dir = os.path.join('/data/eeg', f'{sub}{mon_}', 'eeg.noreref')

    include_pairs_mask = np.empty(len(pairs), dtype=bool)

    if np.char.endswith(eegfile, '.h5'):
        # The h5 timeseries + bipolar labels are identical for every pair, so
        # read them ONCE (was re-read inside the loop, i.e. one full-file read
        # per pair). Last-match semantics on duplicate labels are preserved:
        # a later index overwrites in the dict exactly as the no-break loop did.
        fname = os.path.join(raw_eeg_dir, eegfile)
        f = h5py.File(fname, 'r')
        data = np.empty_like(f['timeseries'])
        f['timeseries'].read_direct(data)
        ch0_labels = np.empty_like(f['bipolar_info']['ch0_label'])
        f['bipolar_info']['ch0_label'].read_direct(ch0_labels)
        ch1_labels = np.empty_like(f['bipolar_info']['ch1_label'])
        f['bipolar_info']['ch1_label'].read_direct(ch1_labels)
        f.close()

        label_to_idx = {(int(ch0_labels[i]), int(ch1_labels[i])): i
                        for i in range(len(ch0_labels))}
        for iPair, pair in pairs.iterrows():
            contact_1, contact_2 = pair[['contact_1', 'contact_2']]
            iChannel = label_to_idx.get((contact_1, contact_2),
                                        label_to_idx.get((contact_2, contact_1)))
            bp_eeg = data[:, iChannel]
            include_pairs_mask[iPair] = not _trace_is_constant(bp_eeg)

    else:
        # Non-h5: variance is a per-contact property, so memmap + test each
        # unique contact ONCE (adjacent bipolar pairs share contacts; was
        # re-read for every pair the contact appeared in).
        flat_cache: dict[Any, bool] = {}

        def _contact_is_flat(contact: Any) -> bool:
            if contact not in flat_cache:
                path = os.path.join(raw_eeg_dir, f'{eegfile}.{str(contact).zfill(3)}')
                flat_cache[contact] = bool(np.var(np.memmap(path, dtype='int16', mode='r')) == 0)
            return flat_cache[contact]

        for iPair, pair in pairs.iterrows():
            contact_1, contact_2 = pair[['contact_1', 'contact_2']]
            try:
                include_pairs_mask[iPair] = not (_contact_is_flat(contact_1) or _contact_is_flat(contact_2))
            except:
                print(iPair, len(include_pairs_mask))

    pairs = pairs[include_pairs_mask].reset_index(drop=True)
    no_bad_channels_removed = np.sum(~include_pairs_mask)
    
    return pairs, no_bad_channels_removed

def check_data(dfrow: pd.Series, root_dir: str) -> Any:
    data_check = pd.Series({})
    for k in ['pairs', 'regionalizations', 'eeg', 'phase']: data_check[k] = False
        
    pairs, localization, pairs_data_source, mat_pairs_available, long_distance_pairs_count, atlas, contacts_source, data_check['pairs'] = check_pairs(dfrow, root_dir)
    if not data_check['pairs']: return data_check
    data_check['pairs_data_source'] = pairs_data_source
    data_check['mat_pairs_available'] = mat_pairs_available
    data_check['long_distance_pairs_count'] = long_distance_pairs_count
    data_check['atlas'] = atlas
    data_check['contacts_source'] = contacts_source
    data_check['localization'] = len(localization) > 0
    if data_check['localization']: localization.reset_index().to_json(join(root_dir, 'electrode_information', 'localization', f'{ftag(dfrow)}_localization.json'))
        
    eeg, data_check['eeg'], eeg_data_check = check_eeg(dfrow, pairs)
    # eeg is None when check_eeg failed; do NOT touch eeg.shape here — the
    # graceful `if not data_check['eeg']: return` below records it as an
    # eeg-load failure. (A leftover `print(eeg.shape)` here used to crash the
    # worker, silently dropping such sessions to an sr_missing misattribution.)
    for k in eeg_data_check.keys():
        data_check[k] = eeg_data_check[k]

    if not data_check['eeg']: return data_check

    data_check['eeg_pairs_match'] = len(data_check['eeg_channels']) == len(pairs)
    if not data_check['eeg_pairs_match']: return data_check

    pairs, no_bad_channels_removed = remove_constant_channels(dfrow, pairs, data_check['evs_data_source'], data_check['eeg_data_source'])
    data_check['pairs_count'] = len(pairs)
    data_check['no_bad_channels_removed'] = no_bad_channels_removed

    pairs.reset_index(drop=True).to_json(join(root_dir, 'electrode_information', 'pairs', f'{ftag(dfrow)}_pairs.json'))

    phase, data_check['phase'] = check_phase(eeg)
    
    regionalizations, data_check['regionalizations'] = check_regionalizations(pairs, localization)
    if not data_check['regionalizations']: return data_check
    data_check['regionalizations_count'] = len(regionalizations)
    #data_check['nonempty_regionalizations_count'] = len([r for r in regionalizations if r != 'nan'])
    return data_check
    
def load_data_check(dfrow: pd.Series, root_dir: str) -> pd.Series:
    
    dfrow = get_dfrow(list(dfrow.name))
    if os.path.exists(join(root_dir, 'data_check', f'{ftag(dfrow)}_data_check.json')):
        data_check = pd.read_json(join(root_dir, 'data_check', f'{ftag(dfrow)}_data_check.json'), typ='series')
        return data_check

def apply_inclusion_rules(
    sess_list_df: pd.DataFrame, min_sample_rate_hz: float
) -> pd.DataFrame:
    """Set the rule-based `include` flags (in place) and return the frame.

    A session is excluded (include=False) if its native sample rate is missing,
    below `min_sample_rate_hz` (499 keeps the ~499.7 Hz BioSemi sessions while
    dropping genuinely sub-500 Hz recordings), or it is not phase-encoded.
    Each excluded session is also stamped with a first-cause `exclusion_reason`
    (`sr_missing` > `sub_500hz` > `data_quality`) for the exclusion report.
    Session-specific denylists are applied separately by the caller.
    """
    sess_list_df['sr_present'] = ~pd.isna(sess_list_df['sr'])
    sess_list_df['include'] = True
    sess_list_df['exclusion_reason'] = ''

    def _exclude(mask: "pd.Series", reason: str) -> None:
        # first-cause: only stamp sessions not already attributed to a prior rule
        fresh = mask & sess_list_df['exclusion_reason'].eq('')
        sess_list_df.loc[mask, 'include'] = False
        sess_list_df.loc[fresh, 'exclusion_reason'] = reason

    _exclude(~sess_list_df['sr_present'], 'sr_missing')
    _exclude(sess_list_df['sr'] < min_sample_rate_hz, 'sub_500hz')
    _exclude(sess_list_df['phase'].eq(False), 'data_quality')
    return sess_list_df


def build_sess_list_df_data_check(root_dir: str) -> pd.DataFrame:
    """
    Loads sess_list_df_initial.csv, attaches load_data_check() outputs, computes include flags,
    and writes sess_list_df_data_check.json in root_dir
    """
    from project_paths import MIN_SAMPLE_RATE_HZ

    sess_list_df = pd.read_csv(join(root_dir, 'sess_list_df_initial.csv'))
    sess_list_df.set_index(['sub', 'exp', 'sess', 'loc', 'mon'], inplace=True, drop=False)

    sess_list_df_data_check = sess_list_df.apply(lambda r: load_data_check(r, root_dir), axis=1)
    sess_list_df[sess_list_df_data_check.columns] = sess_list_df_data_check

    sess_list_df = apply_inclusion_rules(sess_list_df, MIN_SAMPLE_RATE_HZ)

    for r in [('R1093J', 'FR1', 0, 0, 1),
              ('R1331T', 'FR1', 0, 0, 0),
              ('CH042', 'pyFR', 2, 0, 0),
              ('R1277J', 'FR1', 0, 0, 1),
              ('FR140', 'pyFR', 1, 0, 0),
              ('FR160', 'pyFR', 1, 0, 0),
              ('FR280', 'pyFR', 0, 0, 0),
              ('UP001', 'pyFR', 3, 0, 0),
              ('R1216E', 'FR1', 0, 0, 1),
              ('R1216E', 'FR1', 1, 0, 1),
              ('R1235E', 'catFR1', 0, 0, 0),
              ('R1626S', 'catFR1', 8, 0, 0)]:
        if r in sess_list_df.index:
            if sess_list_df.at[r, 'exclusion_reason'] == '':
                sess_list_df.at[r, 'exclusion_reason'] = 'denylist'
            sess_list_df.at[r, 'include'] = False

    for r in [('R1100D', 'FR1', 0, 0, 0),
              ('R1100D', 'FR1', 1, 0, 0),
              ('R1408N', 'catFR1', 0, 0, 0),
              ('R1408N', 'catFR1', 1, 0, 0),
              ('R1275D', 'FR1', 3, 0, 0),
              ('R1310J', 'catFR1', 1, 0, 0),
              ('R1486J', 'catFR1', 4, 0, 1),
              ('R1486J', 'catFR1', 5, 0, 1),
              ('R1486J', 'catFR1', 6, 0, 1),
              ('R1486J', 'catFR1', 7, 0, 1)]:
        if r in sess_list_df.index:
            if sess_list_df.at[r, 'exclusion_reason'] == '':
                sess_list_df.at[r, 'exclusion_reason'] = 'denylist'
            sess_list_df.at[r, 'include'] = False

    # Per-session denylist of empirically-unrecoverable sessions. Curated by
    # the error-triage process in `temp/electrode_coverage/` and other
    # forensic analyses of full-pipeline runs. Anything listed here is
    # force-excluded so the FC stage never attempts to load it. Add new
    # entries via the columns: subject, experiment, session, localization,
    # montage, error, explanation (the last two are bookkeeping, not used
    # for matching).
    from project_paths import UNRECOVERABLE_SESSIONS_CSV
    if UNRECOVERABLE_SESSIONS_CSV.exists():
        unrec = pd.read_csv(UNRECOVERABLE_SESSIONS_CSV)
        if len(unrec):
            for _, row in unrec.iterrows():
                key = (row['subject'], row['experiment'], int(row['session']),
                       int(row['localization']), int(row['montage']))
                if key in sess_list_df.index:
                    if sess_list_df.at[key, 'exclusion_reason'] == '':
                        sess_list_df.at[key, 'exclusion_reason'] = 'unrecoverable'
                    sess_list_df.at[key, 'include'] = False

    # Manual exclusion list (config/excluded_sessions.csv): curatorial removals
    # applied HERE (not after events) so the events/FC stages skip them via
    # include==False and they never enter the event log unaccounted. Reason is
    # stamped 'manual' (override) — it's the operative decision regardless of
    # any data-quality issue.
    excluded_csv = join(os.path.dirname(os.path.abspath(__file__)),
                        'config', 'excluded_sessions.csv')
    if ex(excluded_csv):
        manual = pd.read_csv(excluded_csv)
        if len(manual):
            for _, row in manual.iterrows():
                key = (row['sub'], row['exp'], int(row['sess']),
                       int(row['loc']), int(row['mon']))
                if key in sess_list_df.index:
                    sess_list_df.at[key, 'exclusion_reason'] = 'manual'
                    sess_list_df.at[key, 'include'] = False

    # Any remaining excluded session with no stamped reason -> catch-all, so the
    # exclusion report's per-reason tags always partition data_check_excluded.
    orphan = (~sess_list_df['include'].astype(bool)) & sess_list_df['exclusion_reason'].eq('')
    sess_list_df.loc[orphan, 'exclusion_reason'] = 'data_check_other'

    sess_list_df.to_json(join(root_dir, 'sess_list_df_data_check.json'))

    return sess_list_df
