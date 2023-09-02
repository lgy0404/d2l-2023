# 导入需要的库
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# 加载数据
train_data = TabularDataset('train.csv')
# print(train_data)

# 记录Id列和Sold Price列
id, label = 'Id', 'Sold Price'  # 修复拼写错误

# 数据预处理
large_val_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price', 'Last Sold Price']
for c in large_val_cols + [label]:
    train_data[c] = np.log(train_data[c] + 1)

# 训练
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))

# 预测
test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns=[id]))  # 使用predict函数进行预测
submission = pd.DataFrame({id: test_data[id], label: preds})  # 修复Dataframe拼写错误
submission.to_csv('submission.csv', index=False)