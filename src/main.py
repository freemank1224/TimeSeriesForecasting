import pandas as pd

from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from model import *

if __name__ == "__main__":
    x = 10
    test_MSE ,test_RMSE ,test_MAE,test_R2, = [], [], [],[]
    for i in range(x):
        #加载数据集
        dataset, scaler = load_dataset(config.dataset_path,i, show_data=True)
        # Prepare the dataset for training/testing
        subsequences = extract_subsequences(dataset, lag=config.lag)
        # Split the dataset into train/test set
        train_loader, valid_set, test_set = train_test_split(subsequences)

        #训练模式
        if config.mode == "train":
            # Create new instance of the RNN
            net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1],hidden_dim=config.hidden_dim,num_layers=config.num_layers)
            net.to(device)
            #训练模型
            train_loop(net, config.epochs, config.lr, config.wd,train_loader, valid_set, i,debug=True)
        else:
            # Create new instance of the RNN using default values
            net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1],hidden_dim=parser.get_default('hidden_dim'),num_layers=parser.get_default('num_layers'))
            net.to(device)
            # Load pretrained weights
            net.load_state_dict(torch.load(config.pretrained_path, map_location=device))

        # Display the prediction next to the target output values
        test_MSE,test_RMSE,test_MAE,test_R2=show_evaluation(net,test_set,scaler,i,test_MSE,test_RMSE,test_MAE,test_R2,debug=True)
    df=pd.DataFrame({

        'Test MSE':test_MSE,

        'Test RMSE':test_RMSE,

        'Test MAE':test_MAE,

        'Test R2':test_R2
    })
    df.to_csv('C:/Users/cx/Desktop/zhibiao(144).csv',index=False)