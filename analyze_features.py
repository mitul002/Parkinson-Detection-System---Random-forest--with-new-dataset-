import pandas as pd
import numpy as np

df = pd.read_csv('spiral_feature_dataset.csv')

features = ['Mean_Thickness', 'Std_Thickness', 'Loop_Variation', 'Path_RMSE', 'Path_MAE', 
            'Jerkiness', 'Entropy', 'Curv_Mean', 'Curv_Std']

print('Feature Discrimination Analysis (Healthy vs Parkinson):')
print('='*80)
print(f"{'Feature':<20} | {'Healthy Mean±Std':<20} | {'Parkinson Mean±Std':<20} | {'% Diff'}")
print('-'*80)

for feat in features:
    h = df[df['Label']=='Healthy'][feat]
    p = df[df['Label']=='Parkinson'][feat]
    
    h_mean, p_mean = h.mean(), p.mean()
    h_std, p_std = h.std(), p.std()
    
    # Calculate percentage difference
    if (h_mean + p_mean) != 0:
        diff_pct = abs(h_mean - p_mean) / ((h_mean + p_mean) / 2) * 100
    else:
        diff_pct = 0
        
    print(f'{feat:<20} | {h_mean:7.2f} ± {h_std:5.2f}     | {p_mean:7.2f} ± {p_std:5.2f}     | {diff_pct:5.1f}%')

print('\n' + '='*80)
print('KEY INSIGHT: Higher % Diff = Better discriminative power')
print('The model prioritizes features that differ most between classes!')
