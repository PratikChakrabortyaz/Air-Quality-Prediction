import torch
from torch.utils.data import DataLoader
from data_preprocessing import load_and_preprocess_data
from lstm_model import LSTMModel, AQIDataset, train_lstm, evaluate_lstm
from transformer_model import TransformerModel, train_transformer, evaluate_transformer
from gru_model import GRUModel, train_gru, evaluate_gru
from xgboost_model import train_and_evaluate_xgboost
from deep_rl_model import PPO, AQIEnv, train_ppo, evaluate_ppo
from tcnn_model import TCNNModel, train_tcnn, evaluate_tcnn


df, features, target = load_and_preprocess_data("/kaggle/input/city-hour/city_hour.csv")


train_dataset_lstm = AQIDataset(df.iloc[:int(0.7 * len(df))][features + [target]].values, 24)
test_dataset_lstm = AQIDataset(df.iloc[int(0.85 * len(df)):][features + [target]].values, 24)
train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=64, shuffle=True)
test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=64, shuffle=False)

train_dataset_transformer = AQIDataset(df.iloc[:int(0.7 * len(df))][features + [target]].values, 72)
test_dataset_transformer = AQIDataset(df.iloc[int(0.85 * len(df)):][features + [target]].values, 72)
train_loader_transformer = DataLoader(train_dataset_transformer, batch_size=64, shuffle=True)
test_loader_transformer = DataLoader(test_dataset_transformer, batch_size=64, shuffle=False)

train_dataset = AQIDataset(df.iloc[:int(0.7 * len(df))][features + [target]].values, 36)
test_dataset = AQIDataset(df.iloc[int(0.85 * len(df)):][features + [target]].values, 36)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


print("\n============================")
print("Training LSTM Model...")

lstm_model = LSTMModel(len(features), 128, 1, 3, 0.3).to('cuda')  
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
train_lstm(lstm_model, train_loader_lstm, lstm_optimizer)
evaluate_lstm(lstm_model, test_loader_lstm)


print("\n============================")
print("Training Transformer Model...")

transformer_model = TransformerModel(
    input_dim=len(features), 
    hidden_dim=256, 
    output_dim=1, 
    num_layers=3, 
    num_heads=4, 
    dropout=0.2
).to('cuda')  

transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)
train_transformer(transformer_model, train_loader_transformer, transformer_optimizer)
evaluate_transformer(transformer_model, test_loader_transformer)


print("\n============================")
print("Training GRU Model...")

gru_model = GRUModel(len(features), 128, 1, 3, 0.3).to('cuda')
gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
train_gru(gru_model, train_loader, gru_optimizer)
evaluate_gru(gru_model, test_loader)


print("\n============================")
print("Training XGBoost Model...")

train_and_evaluate_xgboost(df, features, target)


print("\n============================")
print("Training TCNN Model...")

tcnn_model = TCNNModel(
    input_dim=len(features), 
    output_dim=1, 
    num_channels=[64, 64, 64]
).to('cuda')

tcnn_optimizer = torch.optim.Adam(tcnn_model.parameters(), lr=0.001)
train_tcnn(tcnn_model, train_loader, tcnn_optimizer)
evaluate_tcnn(tcnn_model, test_loader)


print("\n============================")
print("Training PPO Model...")

ppo_model = PPO(len(features)).to('cuda')
ppo_optimizer = torch.optim.Adam(ppo_model.parameters(), lr=0.001)

train_env = AQIEnv(df.iloc[:int(0.7 * len(df))][features + [target]].values, 36)
test_env = AQIEnv(df.iloc[int(0.85 * len(df)):][features + [target]].values, 36)

train_ppo(train_env, ppo_model, ppo_optimizer)
evaluate_ppo(test_env, ppo_model)

print("\n============================")
print("All Models Trained and Evaluated Successfully")
