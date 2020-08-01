import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
	# Уберем самые популярные товары (их и так купят)
	popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
	popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

	top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
	data = data[~data['item_id'].isin(top_popular)]

	# Уберем самые НЕ популярные товары (их и так НЕ купят)
	top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
	data = data[~data['item_id'].isin(top_notpopular)]

	# Уберем товары, которые не продавались за последние 12 месяцев
	last_purchase_week = data.groupby('item_id')['week_no'].max().reset_index()
	bought_in_n_weeks = last_purchase_week[
		last_purchase_week['week_no'] > data['week_no'].max() - 52].item_id.tolist()
	data = data[data['item_id'].isin(bought_in_n_weeks)]

	# Уберем не интересные для рекоммендаций категории (department)
	if item_features is not None:
		department_size = pd.DataFrame(item_features.\
										groupby('department')['item_id'].nunique().\
										sort_values(ascending=False)).reset_index()

		department_size.columns = ['department', 'n_items']
		rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
		items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

		data = data[~data['item_id'].isin(items_in_rare_departments)]


	# Уберем слишком дешевые товары (на них не заработаем).
	data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
	data = data[data['price'] > 1]

	# Уберем слишком дорогие товарыs
	data = data[data['price'] < 50]

	# Возбмем топ по популярности
	popularity = data.groupby('item_id')['quantity'].sum().reset_index()
	popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

	top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
	
	# Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
	data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
	
	# ...

	return data


def _check_price(item_id, df_price, limit=7):
	return df_price.loc[df_price['item_id'] == item_id]['price'].values[0] > limit
	

def _check_purchasings(user_id, item_id, data):
	return len(data.loc[data['user_id'] == user_id].loc[data['item_id'] == item_id]) > 0


def postfilter_items(user_id, recommendations, popular_recs, data, df_price, item_features, N=5):
	def get_cat(item_id):
		return item_features.loc[item_features['item_id'] == item_rec]['sub_commodity_desc'].values[0]
		
	n_remaining = N
	new_remaining = 2
	categories = []
	final_recs = []
	
	if recommendations != 0:
		recommendations_pool = recommendations
		recommendations_pool.extend(popular_recs)
	else:
		recommendations_pool = popular_recs
		
	for item_rec in recommendations_pool:
		if _check_price(item_rec, df_price):
			final_recs.append(item_rec)
			categories.append(get_cat(item_rec))
			break
	
	for item_rec in recommendations_pool:
		if not _check_purchasings(user_id, item_rec, data):
			if item_rec in final_recs:
				new_remaining -= 1
				if new_remaining == 0:
					break
			else:
				if get_cat(item_rec) not in categories:
					final_recs.append(item_rec)
					categories.append(get_cat(item_rec))
					new_remaining -= 1
					if new_remaining == 0:
						break
	
	for item_rec in recommendations_pool:
		if len(final_recs) == N:
			break
		if item_rec not in final_recs and get_cat(item_rec) not in categories:
			final_recs.append(item_rec)
			
	return final_recs
	
	
def prepare_data():
	data = pd.read_csv('./raw_data/retail_train.csv')
	item_features = pd.read_csv('./raw_data/product.csv')
	user_features = pd.read_csv('./raw_data/hh_demographic.csv')
	test_data = pd.read_csv('./raw_data/retail_test1.csv')

	# column processing
	item_features.columns = [col.lower() for col in item_features.columns]
	user_features.columns = [col.lower() for col in user_features.columns]

	item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
	user_features.rename(columns={'household_key': 'user_id'}, inplace=True)
	
	return data, item_features, user_features, test_data


def split_data_2_lvl(data, val_lvl_1_size_weeks=6, val_lvl_2_size_weeks=3):
	# Важна схема обучения и валидации!
	# -- давние покупки -- | -- 6 недель -- | -- 3 недель -- 
	# подобрать размер 2-ого датасета (6 недель) --> learning curve (зависимость метрики recall@k от размера датасета)
	data_train_lvl_1 = data[data['week_no'] < data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)]
	data_val_lvl_1 = data[(data['week_no'] >= data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)) &
						  (data['week_no'] < data['week_no'].max() - (val_lvl_2_size_weeks))]

	data_train_lvl_2 = data_val_lvl_1.copy()  # Для наглядности. Далее мы добавим изменения, и они будут отличаться
	data_val_lvl_2 = data[data['week_no'] >= data['week_no'].max() - val_lvl_2_size_weeks]
	
	return data_train_lvl_1, data_val_lvl_1, data_train_lvl_2, data_val_lvl_2
	
	
def add_new_features_for_lvl_2(data, user_features, item_features, items_emb_df, users_emb_df):
	# Item price
	data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
	
	# Embeddings
	item_features = item_features.merge(items_emb_df, how='left')
	user_features = user_features.merge(users_emb_df, how='left')
	
	# User mean basket price
	basket_mean_price = data.groupby(['user_id', 'basket_id'])['sales_value'].mean()
	user_backet_mean_price = basket_mean_price.groupby('user_id').mean()
	user_features = user_features.merge(user_backet_mean_price, on='user_id', how='left')
	user_features = user_features.rename(columns={'sales_value': 'mean_basket'})

	# User mean week purchases
	baskets_per_week = data.groupby(['user_id', 'week_no'])['basket_id'].count()
	user_baskets_per_week_mean = baskets_per_week.groupby('user_id').mean()
	user_features = user_features.merge(user_baskets_per_week_mean, on='user_id', how='left')
	user_features = user_features.rename(columns={'basket_id': 'mean_baskets_weekly'})
	
	# Item mean price by category
	prices = data.groupby(['item_id'])['price'].mean()
	item_features = item_features.merge(prices, on='item_id', how='left')
	commodity = item_features.groupby(['commodity_desc'])['price'].mean()
	item_features = item_features.merge(commodity, on='commodity_desc', how='left')
	item_features = item_features.rename(columns={'sales_value_x': 'price', 'sales_value_y': 'avg_commodity_price'})
	
	# Item mean weekly number of purchases
	items_weekly = data.groupby(['item_id', 'week_no'])['sales_value'].count()
	items_weekly_mean = items_weekly.groupby(['item_id']).mean()
	item_features = item_features.merge(items_weekly_mean, on='item_id', how='left')
	item_features = item_features.rename(columns={'sales_value': 'weekly_mean'})
	
	# User mean amount of purchases by category
	data = data.merge(item_features[['item_id', 'commodity_desc']], on='item_id', how='left')
	user_commodity_total = data.groupby(['user_id', 'commodity_desc'])['sales_value'].mean()
	data = data.merge(user_commodity_total, on=['user_id', 'commodity_desc'], how='left')
	data = data.rename(columns={'sales_value_y':'avg_commodity_val'})
	
	# User mean number of purchases by category compare to overall average number by category
	user_commodity_cnt_weekly = data.groupby(['user_id', 'commodity_desc', 'week_no'])['item_id'].count()
	user_commodity_mean_weekly = user_commodity_cnt_weekly.groupby(['user_id', 'commodity_desc']).mean()
	overall_commodity_mean = user_commodity_cnt_weekly.groupby(['commodity_desc', 'week_no']).mean()
	overall_commodity_mean_weekly = overall_commodity_mean.groupby(['commodity_desc']).mean()
	data = data.merge(user_commodity_mean_weekly, on=['user_id', 'commodity_desc'], how='left').rename(columns={'item_id_y': 'user_commodity_weekly'})
	data = data.merge(overall_commodity_mean_weekly, on=['commodity_desc'], how='left').rename(columns={'item_id': 'commodity_weekly', 'item_id_x': 'item_id'})
	data['user_commodity_comp'] = data['user_commodity_weekly'] / data['commodity_weekly']
	
	return data, user_features, item_features


def get_items_popular_sorted(data):
	pop_rec = data.groupby('item_id')['quantity'].count().reset_index().sort_values('quantity', ascending=False)
	pop_rec = pop_rec[pop_rec['item_id'] != 999999]
	return pop_rec.item_id.tolist()
