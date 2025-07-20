import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####  SAME PROCESS FROM preprocess_gan

# save scalars + tensors
def prep_save(label_col='target',
                        db_path=r'data\initial_db.db',
                        feature_file='features.pt',
                        label_file='labels.pt',
                        scaler_file='scaler.save'):
    

#connect to initial_db
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM eps_data;", conn)
    conn.close()

# drop ignored columns
    ignore_columns = ['Unnamed: 10', 'SurPrice % Change After EPS (1d)prise %']
    features = df.drop(columns=[label_col] + ignore_columns).values
    labels = df[label_col].values

# standardize features + tensors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save(features_tensor, feature_file)
    torch.save(labels_tensor, label_file)
    joblib.dump(scaler, scaler_file)

    print(f"saved features to {feature_file} /// labels to {label_file} /// scaler to {scaler_file}")



# dataloader
class EPSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# cgan
latent_dim = 32  # latent vector dim
num_classes = 2  # 1 is beat, 0 is non beat

def one_hot(labels, num_classes=num_classes):
    return torch.eye(num_classes)[labels].to(device)

# generator model --> conditioned on labels
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )


    def forward(self, noise, labels):
        labels_onehot = one_hot(labels)
        x = torch.cat([noise, labels_onehot], dim=1)  
        return self.model(x)



# discrimantor
class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features, labels):
        labels_onehot = one_hot(labels)
        x = torch.cat([features, labels_onehot], dim=1)  
        return self.model(x)





#train
def train_cgan(num_epochs=100, batch_size=128, lr=0.0002):
    
    # load data + dataset/dataloader
    features = torch.load('features.pt')
    labels = torch.load('labels.pt')
    feature_dim = features.shape[1]

    
    dataset = EPSDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # start the models (optimizers + loss function)
    G = Generator(latent_dim, num_classes, feature_dim).to(device)
    D = Discriminator(feature_dim, num_classes).to(device)

 
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)




    #  loop
    for epoch in range(num_epochs):
        for real_features, real_labels in dataloader:
            batch_size = real_features.size(0)
            real_features = real_features.to(device)
            real_labels = real_labels.to(device)


            real_targets = torch.ones(batch_size, 1).to(device)
            fake_targets = torch.zeros(batch_size, 1).to(device)

            #discriminator training
            optimizer_D.zero_grad()
            output_real = D(real_features, real_labels)
            loss_real = criterion(output_real, real_targets)


            noise = torch.randn(batch_size, latent_dim).to(device)
            random_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_features = G(noise, random_labels)


            output_fake = D(fake_features.detach(), random_labels)
            loss_fake = criterion(output_fake, fake_targets)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()


            # generator training
            optimizer_G.zero_grad()
            output_fake = D(fake_features, random_labels)
            loss_G = criterion(output_fake, real_targets)
            loss_G.backward()
            optimizer_G.step()

        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

    
    torch.save(G.state_dict(), 'generator.pth')
    print("Saved generator.pth")

#new data 
def new_data(num_samples=1000):
    
    # laoding dataset 
    scaler = joblib.load('scaler.save')
    features = torch.load('features.pt')
    feature_dim = features.shape[1]

    
    G = Generator(latent_dim, num_classes, feature_dim).to(device)
    G.load_state_dict(torch.load('generator.pth')) #load generator files
    G.eval()

    # distribution of beat vs non-beat from original dataset
    labels = torch.load('labels.pt')
    num_beat = (labels == 1).sum().item()
    num_non_beat = (labels == 0).sum().item()

    # the ratio of beat to non-beat samples
    total_samples = num_beat + num_non_beat
    ratio_beat = num_beat / total_samples
    ratio_non_beat = num_non_beat / total_samples

    #  synthetic data for both classes based on the calculated ratio
    num_beat_samples = int(num_samples * ratio_beat)
    num_non_beat_samples = num_samples - num_beat_samples

    new_data = []

    # generate data for label = 1
    noise_beat = torch.randn(num_beat_samples, latent_dim).to(device)
    labels_beat = torch.full((num_beat_samples,), 1, dtype=torch.long).to(device)
    with torch.no_grad():
        synthetic_features_beat = G(noise_beat, labels_beat).cpu().numpy()

    synthetic_features_beat_orig = scaler.inverse_transform(synthetic_features_beat)
    columns = [
        'Shares Outstanding', 'Beta', 'PE Ratio', 'Market Cap', 'EPS Actual', 'Revenue',
        'EPS Estimate', 'EPS Surprise', 'price_pct_change_1d_before', 'price_pct_change_5d_before',
        'volume_pct_change_1d_before', 'volatility_5d', 'ma_5', 'ma_10', 'ma_20'
    ]

    df_synthetic_beat = pd.DataFrame(synthetic_features_beat_orig, columns=columns)
    df_synthetic_beat['target'] = 1
    new_data.append(df_synthetic_beat)

    # generate data for label = 0
    noise_non_beat = torch.randn(num_non_beat_samples, latent_dim).to(device)
    labels_non_beat = torch.full((num_non_beat_samples,), 0, dtype=torch.long).to(device)
    
    
    with torch.no_grad():
        synthetic_features_non_beat = G(noise_non_beat, labels_non_beat).cpu().numpy()

    synthetic_features_non_beat_orig = scaler.inverse_transform(synthetic_features_non_beat)
    df_synthetic_non_beat = pd.DataFrame(synthetic_features_non_beat_orig, columns=columns)
    df_synthetic_non_beat['target'] = 0
    new_data.append(df_synthetic_non_beat)


    full_synthetic_df = pd.concat(new_data, axis=0, ignore_index=True)
    return full_synthetic_df



def sql_save(df, db_path='data/new_data.db'):
 
    conn = sqlite3.connect(db_path)
    df.to_sql('eps_data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"saved new data to {db_path}")



#check if pth file is created
#discrimnator loss increases --> lr decrease

#________________________________________________________________________________________________________

if __name__ == '__main__':
    prep_save(label_col='target')
    train_cgan(num_epochs=1000)
    synthetic_df = new_data(num_samples=50000)
    sql_save(synthetic_df)
