# import sqlite3
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler
# import joblib 

# #paths to save all these files
# def preprocess_and_save(label_col='target', 
#                         db_path=r'data\initial_db.db',
#                         feature_file='features.pt',
#                         label_file='labels.pt',
#                         scaler_file='scaler.save'):  

# #connect to initial_db
#     conn = sqlite3.connect(db_path)
#     df = pd.read_sql_query("SELECT * FROM eps_data;", conn)
#     conn.close()

# #need to drop the previous columns that were dropped
#     ignore_columns = ['Unnamed: 10', 'SurPrice % Change After EPS (1d)prise %']
#     features = df.drop(columns=[label_col] + ignore_columns).values
#     labels = df[label_col].values



# #normalize the data --> mean = 0
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)

    
#     joblib.dump(scaler, scaler_file)
#     print(f"saved    {scaler_file}")

#     features = torch.tensor(features, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.long)



# #save all scalars+tensors
#     torch.save(features, feature_file)
#     torch.save(labels, label_file)
#     print(f"saved features to {feature_file} / labels to {label_file}")

# if __name__ == '__main__':
#     preprocess_and_save(label_col='target')
