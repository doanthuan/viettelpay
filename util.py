import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import pickle

def preprocess_data(train, account, sale, trans_hist, features, cat_features):
    
    # aggregate trans hist data
    train = train.merge(trans_hist, how='left', left_on=['ben_msisdn'], right_on=['msisdn_'])
    
    #aggregate account, sale data
    train = build_RW000076(train, account, sale)
    
    
    train['request_date_dt'] = pd.to_datetime(train['request_date'], format='%Y-%m-%d %H:%M:%S')
    #train['request_date'] = (train['request_date_dt'] - train['request_date_dt'].min()).days
    train['date_diff'] = (train['request_date_dt'] - train['request_date_dt'].min()).dt.days
    
#     train["same_name"] = train["cust_name"]==train["ben_cust_name"]
#     train["same_phone"] = train["msisdn"]==train["ben_msisdn"]
#     train["same_phone_channel"] = train["msisdn"]==train["msisdn_channel"]
#     train["same_phone_channel_ben"] = train["ben_msisdn"]==train["msisdn_channel"]
    
    train[cat_features] = train[cat_features].fillna(value="")
    #train[cat_features].fillna('', inplace=True)
    
    y = train['is_fraud']
    X = train.drop(['is_fraud'], axis = 1)
    X = X[features]
    return X, y


def split_data(data):

    y = data['is_fraud']
    X = data.drop(['is_fraud'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    return X, y, X_train, y_train, X_val, y_val, X_test, y_test


def split_val_data(train):

    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


def eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print('classification_report: \n{}'.format(classification_report(y_test, y_pred)))
    print('confusion_matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))
    print('roc_auc_score: {}'.format(roc_auc_score(y_test, y_pred)))
    print('precision_score: {}'.format(precision_score(y_test, y_pred)))
    print('recall_score: {}'.format(recall_score(y_test, y_pred)))
    print('f1_score: {}'.format(f1_score(y_test, y_pred)))
    

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def build_RW000076(train, account, sale):
    account_sale = sale.merge(account, how='inner', left_on=['account_sale_id', 'phone'], right_on=['account_sale_id', 'msisdn'])
    account_sale = account_sale[["phone","staff_code"]]
    train = train.merge(account_sale, how='left', left_on=['ben_msisdn','staff_code'], right_on=['phone','staff_code'])
    train["RW000076"] = pd.notnull(train["phone"])
    return train
