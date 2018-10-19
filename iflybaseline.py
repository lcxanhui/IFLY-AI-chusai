import numpy as np 
import pandas as pd
import time
import datetime
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import xgboost as xgb
# 加载数据
train = pd.read_table('/Users/pc/Desktop/ifly/round1_iflyad_train.txt')
test = pd.read_table('/Users/pc/Desktop/ifly/round1_iflyad_test_feature.txt')
# 合并训练集，验证集
data = pd.concat([train,test],axis=0,ignore_index=True,sort=True)
# 缺失值填充
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))
# replace
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])
# labelencoder 转化
encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
           'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
           'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

data['day'] = data['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))

# 历史点击率
# 时间转换
data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31

for feat_1 in ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    gc.collect()
    res=pd.DataFrame()
    temp=data[[feat_1,'period','click']]
    for period in range(27,35):
        if period == 27:
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
        else: 
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
        count['period']=period
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    data = pd.merge(data,res, how='left', on=[feat_1,'period'])


# 删除没用的特征
drop = ['click', 'time', 'instance_id', 'user_tags', 
        'app_paid', 'creative_is_js', 'creative_is_voicead']

train = data[:train.shape[0]]
test = data[train.shape[0]:]

y_train = train.loc[:,'click']
res = test.loc[:, ['instance_id']]

train.drop(drop, axis=1, inplace=True)
print('train:',train.shape)
test.drop(drop, axis=1, inplace=True)
print('test:',test.shape)

X_loc_train = train.values
y_loc_train = y_train.values
X_loc_test = test.values

# 模型部分
model = xgb.XGBClassifier(booster='gbtree', max_depth=12, learning_rate=0.05, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary:logistic', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,eval_metric='logloss',
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=2018, n_jobs=-1, silent=True)

# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1000))

for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    xgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                          eval_set=[(X_loc_train[train_index], y_loc_train[train_index]), 
                                    (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=100,verbose=50)
    test_pred= xgb_model.predict_proba(X_loc_test)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred

# 加权平均
res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]       
res['predicted_score'] = res['predicted_score']/5

# 提交结果
mean = res['predicted_score'].mean()
print('mean:',mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res[['instance_id', 'predicted_score']].to_csv("lgb_baseline_%s.csv" % now, index=False)
