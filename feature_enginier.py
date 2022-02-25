import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class feature_engineering:
    def __init__(self, data):
        # replace (-1) values with np.nan
        data.replace(-1, np.nan, inplace=True)
        data.replace('-1', np.nan, inplace=True)

        self.train = True
        self.mean = None
        self.data = data
        self.useless_columns = ['product_category(2)', 'product_category(3)', 'product_category(4)', 'product_category(5)', 'product_category(6)','product_category(7)','product_id', 'product_title', 'user_id', 'time_delay_for_conversion', 'product_price','SalesAmountInEuro']
        self.label_encode = {}
        self.onehot_encode = {}
        self.mean_norm = None
        self.std_norm = None
        self.continues_columns = ['nb_clicks_1week','product_age_group_percentage','partner_id_percentage','audience_id_percentage','brand_percentage']

        self.train_columns = {}
        self.merge_df('product_brand', 40)
        self.merge_df('audience_id', 70)
        self.merge_df('partner_id', 500)
        self.merge_df('product_age_group', 70)
        self.percentage_df('product_country', 1000)
        self.percentage_df('product_gender', 600)

        self.train_columns['device_type'] = self.data['device_type'].unique().tolist()
        self.train_columns['product_category(1)'] = self.data['product_category(1)'].unique().tolist()

    def change_data(self, data, train=False):
        self.data = data
        data.replace(-1, np.nan, inplace=True)
        data.replace('-1', np.nan, inplace=True)
        self.train = train

    # main cleaning function
    def clean_function(self):
        # change format of click timestamp
        self.data['click_timestamp'] = pd.to_datetime(self.data['click_timestamp'], format="%Y-%m-%d %H:%M:%S")
        self.data['hour_of_click'] = self.data['click_timestamp'].dt.hour
        if self.train:
            self.mean = self.data['Sale'].mean()

        self.remove_useless_columns()
        self.click_timestamp()
        self.product_category()
        self.product_country()
        self.product_gender()
        self.product_age_group()
        self.partner_id()
        self.audience_id()
        self.device_type()
        self.product_brand()
        self.nb_click_1week()
        self.normalize()

    # remove useless columns
    def remove_useless_columns(self):
        self.data.drop(columns=self.useless_columns, inplace=True)

    # temp function to determine a corresponding period for a given time
    def number_of_period(self, num):
        if num < 3:
            return 1
        elif num < 6:
            return 2
        elif num < 9:
            return 3
        elif num < 12:
            return 4
        elif num < 15:
            return 5
        elif num < 18:
            return 6
        elif num < 21:
            return 7
        else:
            return 8

            # returns one hot encoded columns

    def one_hot_encode(self, values, name):
        if self.train:
            # integer encode
            self.label_encode[name] = LabelEncoder()
            self.label_encode[name].fit(values)
            integer_encoded = self.label_encode[name].transform(values)

            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoder.fit(integer_encoded)
            self.onehot_encode[name] = onehot_encoder

            # integer encode
        integer_encoded = self.label_encode[name].transform(values)
        # binary encode
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encode[name].transform(integer_encoded)
        # transpose one_hot encoded data
        onehot_encoded = np.array(onehot_encoded).transpose()
        return onehot_encoded

    def unique_values(self, name):
        df1 = self.data[name].unique().tolist()
        df1 = [str(x) for x in df1]
        df1 = [x for x in df1 if x != 'nan']
        return df1

    def convert_to_onehot(self, name, new_name):
        df1 = self.unique_values(name)
        not_mentioned_categories = list(set(df1) - set(self.train_columns[name]))
        self.data[name].replace(to_replace=not_mentioned_categories, value=np.nan, inplace=True)
        values = np.array(self.data[name].tolist())
        onehot_encoded = self.one_hot_encode(values, name)
        for i, column in enumerate(onehot_encoded):
            self.data.insert(1, f'{new_name} {i + 1}', column, True)
        self.data.drop(columns=[name], inplace=True)

    # encode click_timestamp to one hot columns
    def click_timestamp(self):
        # divide timestamp into 8 periods
        self.data['click_timestamp'] = self.data['hour_of_click'].apply(lambda x: self.number_of_period(int(x)))
        # add onehot_encoded results into main dataframe
        values = np.array(self.data['click_timestamp'].tolist())
        onehot_encoded = self.one_hot_encode(values, 'click_timestamp')
        for i, column in enumerate(onehot_encoded):
            self.data.insert(1, f'period {i + 1}', column, True)
        self.data.drop(columns=['click_timestamp', 'hour_of_click'], inplace=True)

        # encode product category to one hot columns

    def product_category(self):
        self.convert_to_onehot('product_category(1)', 'sub_category')

    # encode device type to one hot vectors
    def product_country(self):
        self.convert_to_onehot('product_country', 'country')

    # encode device type to one hot vectors
    def device_type(self):
        self.convert_to_onehot('device_type', 'device_type')

    # encode product gender to one hot vectors
    def product_gender(self):
        self.convert_to_onehot('product_gender', 'gender')

    # assign corresponding sale percentage to given feature
    def assign_percentage(self, name, column, df):
        df.drop(columns=['Sale_x'], inplace=True)
        # replace NAN values with mean
        df.replace(np.nan, self.mean, inplace=True)
        self.data.drop(columns=[column], inplace=True)
        self.data.insert(len(self.data.columns), name, df['Sale_y'].tolist(), True)

    # assign corresponding sale percentage to each product band
    def product_brand(self):
        df = pd.merge(self.data[['product_brand', 'Sale']], self.train_columns['product_brand'], on='product_brand',
                      how='left')
        self.assign_percentage('brand_percentage', 'product_brand', df)

        # assign corresponding sale percentage to each audience id

    def audience_id(self):
        df = pd.merge(self.data[['audience_id', 'Sale']], self.train_columns['audience_id'], on='audience_id',
                      how='left')
        self.assign_percentage('audience_id_percentage', 'audience_id', df)

    # assign corresponding sale percentage to each partner id
    def partner_id(self):
        df = pd.merge(self.data[['partner_id', 'Sale']], self.train_columns['partner_id'], on='partner_id', how='left')
        self.assign_percentage('partner_id_percentage', 'partner_id', df)

        # assign corresponding sale percentage to each age group

    def product_age_group(self):
        df = pd.merge(self.data[['product_age_group', 'Sale']], self.train_columns['product_age_group'],
                      on='product_age_group', how='left')
        self.assign_percentage('product_age_group_percentage', 'product_age_group', df)

    def nb_click_1week(self):
        self.data['nb_clicks_1week'].replace(np.nan, self.mean, inplace=True)

    def normalize(self):
        if self.train:
            self.mean_norm = self.data[self.continues_columns].mean()
            self.std_norm = self.data[self.continues_columns].std()
        self.data[self.continues_columns] = (self.data[self.continues_columns] - self.mean_norm) / self.std_norm

    def merge_df(self, name, limit):
        product_feature_and_sale = self.data[[name, 'Sale']].groupby(name).filter(lambda x: len(x) > limit).groupby(
            name).mean()
        feature_groups = product_feature_and_sale.index.tolist()
        sales = []
        for i in range(len(feature_groups)):
            sales.append(product_feature_and_sale['Sale'][i])
        feature_sale = pd.DataFrame({name: feature_groups, 'Sale': sales})
        self.train_columns[name] = feature_sale


    def percentage_df(self, name,limit):
        product_feature_and_sale = self.data[[name, 'Sale']].groupby(name).filter(lambda x: len(x) > limit).groupby(name).mean()
        product_features = product_feature_and_sale.index.tolist()
        secondary_features = list(set(product_features))
        self.train_columns[name] = secondary_features