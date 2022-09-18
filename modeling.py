import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def LGBMClassifierAuto(X_train,
                       y_train,
                       X_test=pd.DataFrame(),
                       col_id=None,
                       plot_roc_curve=True,
                       plot_learning_curve=True,
                       plot_feature_importance=True,
                       plot_decision_tree=True,
                       seed=2022):

    validations = pd.DataFrame()
    predictions = pd.DataFrame()
    models = []

    cols_rm = [col_id]
    X_test_ = pd.DataFrame()
    if (col_id!=None)&(len(X_test)!=0):
        X_test_ = X_test.drop(cols_rm,axis=1)

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_id, valid_id in skf.split(X_train,y_train):
        if col_id==None:
            X_train_ = X_train.iloc[train_id]
            X_valid_ = X_train.iloc[valid_id]
        else:
            X_train_ = X_train.drop(cols_rm,axis=1).iloc[train_id]
            X_valid_ = X_train.drop(cols_rm,axis=1).iloc[valid_id]
        y_train_ = y_train.iloc[train_id]
        y_valid_ = y_train.iloc[valid_id]

        model = lgbm.LGBMClassifier(boosting_type='gbdt',
                               num_leaves=31,
                               max_depth=-1,
                               learning_rate=0.1,
                               n_estimators=100,
                               reg_alpha=0.0,
                               reg_lambda=0.0,
                               random_state=seed)
        model.fit(X_train_, y_train_, eval_set=[(X_valid_, y_valid_), (X_train_, y_train_)], callbacks=[lgbm.log_evaluation(0)])
        models.append(model)

        validations = pd.concat([
            validations,
            pd.DataFrame({
                "prediction":model.predict_proba(X_valid_).T[1],
                "actual":y_valid_
            })
        ])

        if len(X_test_)!=0:
            predictions = pd.concat([
                predictions,
                pd.DataFrame({
                    "id":X_test[col_id],
                    "prediction":model.predict_proba(X_test_).T[1]
                })
            ])

    # prediction
    if len(X_test_)!=0:
        predictions = predictions.groupby("id",as_index=False).mean()

    # ROC
    if plot_roc_curve:
        fpr, tpr, thresholds = metrics.roc_curve(validations["actual"], validations["prediction"])
        roc_auc = metrics.auc(fpr, tpr)
        roc_plt = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
        roc_plt.plot()
        plt.show()

    # Evaluation
    if plot_learning_curve:
        figure, ax = plt.subplots(
            1,
            len(models),
            constrained_layout=True,
            figsize=(20,4),
        )
        for i, model in enumerate(models):
            lgbm.plot_metric(model,ax=ax[i],title=f"model {i}")
        plt.show()

    # PFI
    if plot_feature_importance:
        figure, ax = plt.subplots(
            1,
            len(models),
            constrained_layout=True,
            figsize=(25,5),
            dpi=200
        )
        for i, model in enumerate(models):
            lgbm.plot_importance(model,ax=ax[i],title=f"model {i}")
        plt.show()

    # Decision Tree
    if plot_decision_tree:
        for i, model in enumerate(models):
            print(f"model {i}")
            lgbm.plot_tree(model,figsize=(5,4),dpi=200)
            plt.show()

    return models, predictions
