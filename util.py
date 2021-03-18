import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import pickle

def preprocess_data(train_df, account_df, sale_df, features, cat_features):
    train_df = build_RW000076(train_df, account_df, sale_df)
    data['request_date_dt'] = pd.to_datetime(data['request_date'], format='%Y-%m-%d %H:%M:%S')
    #data['request_date'] = (data['request_date_dt'] - data['request_date_dt'].min()).days
    data['date_diff'] = (data['request_date_dt'] - data['request_date_dt'].min()).dt.days
    
    data["same_name"] = data["cust_name"]==data["ben_cust_name"]
    data["same_phone"] = data["msisdn"]==data["ben_msisdn"]
    data["same_phone_channel"] = data["msisdn"]==data["msisdn_channel"]
    data["same_phone_channel_ben"] = data["ben_msisdn"]==data["msisdn_channel"]
    
    data[cat_features] = data[cat_features].fillna(value="")
    #data[cat_features].fillna('', inplace=True)
    
    
    
    y = data['is_fraud']
    X = data.drop(['is_fraud'], axis = 1)
    X = X[features]
    return X, y


def split_data(data):

    y = data['is_fraud']
    X = data.drop(['is_fraud'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    return X, y, X_train, y_train, X_val, y_val, X_test, y_test


def split_val_data(train_df):

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

def build_RW000076(train_df, account_df, sale_df):
    account_sale_df = sale_df.merge(account_df, how='inner', left_on=['account_sale_id', 'phone'], right_on=['account_sale_id', 'msisdn'])
    account_sale_df = account_sale_df[["phone","staff_code"]]
    train_df = train_df.merge(account_sale_df, how='left', left_on=['ben_msisdn','staff_code'], right_on=['phone','staff_code'])
    train_df["RW000076"] = pd.notnull(train_df["phone"])
    return train_df