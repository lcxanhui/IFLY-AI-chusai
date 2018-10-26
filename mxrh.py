model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)


skf=list(StratifiedKFold(y_loc_train, n_folds=10, shuffle=True, random_state=1024))
for i, (train, test) in enumerate(skf):
    print("Fold", i)
    model.fit(X_loc_train[train], y_loc_train[train], eval_metric='logloss',eval_set=[(X_loc_train[train], y_loc_train[train]), (X_loc_train[test], y_loc_train[test])],early_stopping_rounds=100)
    
    test_pred= model.predict_proba(X_loc_test, num_iteration=-1)[:, 1]
    print('test mean:', test_pred.mean())
    
    res['prob_%s' % str(i)] = test_pred
    res[['instance_id', 'prob_%s' % str(i)]].to_csv('prob_%s' % str(i), index=False, sep=" ")

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')

res['predicted_score'] = 0

# 加权平均融合, 当模型效果差异较大时，对结果进行加权平均；线上效果好的权重相对大些，而线上效果差的权重相对小些。
# for i in range(10):
#     res['predicted_score'] += res['prob_%s' % str(i)]
# res['predicted_score'] = res['predicted_score']/10

# 反smigod融合,当模型的效果差异比较小时采取此方法
for i in range(10):
    res['predicted_score'] += res['prob_%s' % str(i)].apply(lambda x: math.log(x/(1-x)))
res['predicted_score'] = (res['predicted_score']/10).apply(lambda x: 1/(1+math.exp(-x)))

mean = res['predicted_score'].mean()
print('mean:',mean)
res[['instance_id', 'predicted_score']].to_csv("./submit/lgbEnsemble_%s.csv" % now, index=False, sep=" ")

t_end = datetime.datetime.now()
print('training time: %s' % ((t_end - t_start).seconds/60))
