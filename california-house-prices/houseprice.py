# 导入需要的库
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# 加载数据
train_data = TabularDataset('train.csv')
# print(train_data)

# 记录Id列和Sold Price列
id, lable = 'Id','Sold Price'

# 数据预处理
large_val_cols = ['Lot', 'Total interior livable area','Tax assessed value', 'Annual tax amount','Listed Price','Last Sold Price']
for c in large_val_cols + [lable]:
    train_data[c] = np.log(train_data[c] + 1)
    
# 训练
predictor = TabularPredictor(label = lable).fit(train_data.drop(columns = [id])) 

# 预测
test_data = TabularDataset('test.csv')
preds = predictor.leaderboard(test_data.drop(columns = [id]))
submission = pd.Dataframe({id:test_data[Id], lable:preds})
submission.to_csv('submission.csv', index = False)