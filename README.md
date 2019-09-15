#仅用于学习日记，不可t用于任何商业用途
#基于LR（逻辑回归）的预估
#1、整理数据  利用字典将数据中的非数字用数字表示
#2、分割数据  sklearn.model_selection.train_test_split
#3、定义模型和训练
#sklearn.linear_model.LogisticRegression(penalty,tol,max_items,fit_intercept) 选择正则化、迭代次数、最小误差、截距
#lr.fit(x_train,y_train)
#4、评估
#lr.predict(x_test) 逻辑回归自带sigmoid 预测出来直接是0/1
#lr.predict_prob(x_test) 这个预测出来是概率 需换为0/1
#sklearn.metrics.mean_squared_error  mse预估
#sklearn.metrics.accuracy_score  正确率评估
