import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import pickle

def preprocess_data(train_df, account_df, sale_df, features, cat_features):
    train_df = build_RW000076(train_df, account_df, sale_df)
    train_df['request_date_dt'] = pd.to_datetime(train_df['request_date'], format='%Y-%m-%d %H:%M:%S')
    #train_df['request_date'] = (train_df['request_date_dt'] - train_df['request_date_dt'].min()).days
    train_df['date_diff'] = (train_df['request_date_dt'] - train_df['request_date_dt'].min()).dt.days
    
    train_df["same_name"] = train_df["cust_name"]==train_df["ben_cust_name"]
    train_df["same_phone"] = train_df["msisdn"]==train_df["ben_msisdn"]
    train_df["same_phone_channel"] = train_df["msisdn"]==train_df["msisdn_channel"]
    train_df["same_phone_channel_ben"] = train_df["ben_msisdn"]==train_df["msisdn_channel"]
    
    train_df[cat_features] = train_df[cat_features].fillna(value="")
    #train_df[cat_features].fillna('', inplace=True)
    
    
    
    y = train_df['is_fraud']
    X = train_df.drop(['is_fraud'], axis = 1)
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