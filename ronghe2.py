import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
path = './data'

warnings.filterwarnings("ignore")

train = pd.read_table(path + '/round1_iflyad_train.txt')
test = pd.read_table(path + '/round1_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)

data = data.fillna(-1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
data['area'] = data['creative_height'] * data['creative_width']


bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)

data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])

data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31


for feat_1 in ['advert_id','advert_industry_inner_1', 'advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel','area']:
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


ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump' ,'advert_id_rate','advert_industry_inner_1_rate','advert_industry_inner_rate', 'advert_name_rate',
			'campaign_id_rate','creative_height_rate','creative_tp_dnf_rate','creative_width_rate' ,'province_rate', 'f_channel_rate']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

count_feature=['cnt_click_of_adid', 'cnt_click_of_advert_id',
       'cnt_click_of_campaign_id', 'cnt_click_of_creative_id',
       'cnt_click_of_os', 'cnt_click_of_carrier']
count_and_feature=['cnt_click_of_advert_id_and_adid', 'cnt_click_of_campaign_id_and_adid',
       'cnt_click_of_creative_id_and_adid', 'cnt_click_of_os_and_adid',
       'cnt_click_of_carrier_and_adid',
       'cnt_click_of_campaign_id_and_advert_id',
       'cnt_click_of_creative_id_and_advert_id',
       'cnt_click_of_os_and_advert_id', 'cnt_click_of_carrier_and_advert_id',
       'cnt_click_of_creative_id_and_campaign_id',
       'cnt_click_of_os_and_campaign_id',
       'cnt_click_of_carrier_and_campaign_id',
       'cnt_click_of_os_and_creative_id',
       'cnt_click_of_carrier_and_creative_id', 'cnt_click_of_carrier_and_os']

cate_feature = origin_cate_list+count_feature+count_and_feature

num_feature = ['creative_width', 'creative_height', 'hour'  , 'area', 'period', 'area_rate']

feature = cate_feature + num_feature
print(len(feature), feature)

predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

del data['click']
# 默认加载 如果 增加了cate类别特征 请改成false重新生成
if os.path.exists(path + '/feature/base_train_csr.npz') and True:
    print('load_csr---------')
    base_train_csr = sparse.load_npz(path + '/feature/base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz(path + '/feature/base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=10)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz(path + '/feature/base_train_csr.npz', base_train_csr)
    sparse.save_npz(path + '/feature/base_predict_csr.npz', base_predict_csr)

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)
feature_select = SelectPercentile(chi2, percentile=50)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
predict_csr = feature_select.transform(predict_csr)
print('feature select')
print(train_csr.shape)

n = 1500
data_col=pd.read_csv('col_sort_one11.csv',header = None)
col=data_col[0].values.copy()


lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=60, max_depth=-1, learning_rate=0.1, n_estimators=n,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=1, reg_lambda=1, seed=2018, nthread=10, silent=True)

skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    print("Fold", index)
    
    i=1300
    lgb_model.fit(train_csr[train_index][:,col[:i]], train_y[train_index],
                  eval_set=[(train_csr[train_index][:,col[:i]], train_y[train_index]),
                            (train_csr[test_index][:,col[:i]], train_y[test_index])], early_stopping_rounds=100,verbose=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr[:,col[:i]], num_iteration=lgb_model.best_iteration_)[:, 1]   
    print('test mean:', test_pred.mean())

    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score']/5
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv(path + "lgb_baseline_%s.csv" % now, index=False)
