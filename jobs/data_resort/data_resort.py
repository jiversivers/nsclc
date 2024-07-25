import glob
import os
import pandas as pd
import re

source = ['E:\\NSCLC_Data_for_ML',  # FOV sorted
          'E:\\NSCLC_intensity_data']  # Subject sorted
destination = r'E:\NSCLC_full_stack_for_ML'
features = pd.read_excel(f'{source[0]}{os.sep}Features_NSCLC_StageII.xlsx')

files_wanted = {'RawRedoxMap.tiff': 'orr.tiff',
                'ROI_Mask.tiff': 'mask.tiff',
                '*_a1.asc': 'a1.asc',
                '*_a2.asc': 'a2.asc',
                '*_t1.asc': 't1.asc',
                '*_t2.asc': 't2.asc',
                '*phasor_G.asc': 'G.asc',
                '*phasor_S.asc': 'S.asc',
                '*photons.asc': 'photons.asc',
                'BlueChNorm.tiff': 'nadh.tiff',
                'GreenChNorm.tiff': 'fad.tiff',
                'UVChNorm.tiff': 'shg.tiff'
                }
for subject in features['Subject ID']:
    os.makedirs(f'{destination}{os.sep}{subject}', exist_ok=True)
match = []
for subject, slide in zip(features['Subject ID'], features['Slide Name']):
    subject_pattern = rf'.*{slide}.*(?P<sample>sample\dof\d).*(?P<fov>fov\d)'
    regex = re.compile(subject_pattern)
    for folder in glob.glob(f'{source[0]}{os.sep}*'):
        match.append(re.match(regex, folder))
        if match[-1]:
            new_folder = f'{destination}{os.sep}{subject}{os.sep}{match[-1].group('sample')}{os.sep}{match[-1].group('fov')}'
            os.makedirs(new_folder, exist_ok=True)

            for current, clean in files_wanted.items():
                f = glob.glob(f'{folder}{os.sep}FLIM{os.sep}{current}')
                if f:
                    os.system(f'copy {f[0]} {new_folder}{os.sep}{clean}')
                f = glob.glob(f'{folder}{os.sep}Redox{os.sep}{current}')
                if f:
                    os.system(f'copy {f[0]} {new_folder}{os.sep}{clean}')
for rt, dr, f in os.walk(source[1]):
    if f:
        regex = re.compile(r'.*subject_(?P<subject>.*)\\.*(?P<sample>sample\dof\d).*(?P<fov>fov\d)')
        match = re.match(regex, rt)
        new_folder = f'{destination}{os.sep}{match.group('subject')}{os.sep}{match.group('sample')}{os.sep}{match.group('fov')}'
        if '755' in rt:
            os.system(f'copy {rt}{os.sep}BlueChNorm.tiff {new_folder}{os.sep}{files_wanted['BlueChNorm.tiff']}')
        if '855' in rt:
            os.system(f'copy {rt}{os.sep}GreenChNorm.tiff {new_folder}{os.sep}{files_wanted['GreenChNorm.tiff']}')
            os.system(f'copy {rt}{os.sep}UVChNorm.tiff {new_folder}{os.sep}{files_wanted['UVChNorm.tiff']}')
for rt, dr, f in os.walk(source[1]):
    if f:
        regex = re.compile(r'.*subject_(?P<subject>.*)\\.*(?P<sample>sample\dof\d).*(?P<fov>fov\d)')
        match = re.match(regex, rt)
        new_folder = f'{destination}{os.sep}{match.group('subject')}{os.sep}{match.group('sample')}{os.sep}{match.group('fov')}'
        if '755' in rt:
            os.system(f'copy {rt}{os.sep}BlueChNorm.tiff {new_folder}{os.sep}{files_wanted['BlueChNorm.tiff']}')
        if '855' in rt:
            os.system(f'copy {rt}{os.sep}GreenChNorm.tiff {new_folder}{os.sep}{files_wanted['GreenChNorm.tiff']}')
            os.system(f'copy {rt}{os.sep}UVChNorm.tiff {new_folder}{os.sep}{files_wanted['UVChNorm.tiff']}')