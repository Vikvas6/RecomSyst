import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, item_features, weighting=True, n_factors=50):
                
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = dict(zip(item_features["item_id"], item_features["brand"] == "Private"))
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.n_factors = n_factors
        self.model = self.fit(self.user_item_matrix, n_factors=n_factors)
     
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват ьдругие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_item(self, item_id, filter_ctm=True):
        if filter_ctm:
            rec = []
            similar_items = self.model.similar_items(self.itemid_to_id[item_id], N=20)[1:]
            for item in similar_items:
                cur_item_id = self.id_to_itemid[item[0]]
                if self.item_id_to_ctm[cur_item_id]:
                     return cur_item_id
        rec = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        return self.id_to_itemid[rec[1][0]]

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        top_user_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        rec = top_user_purchases['item_id'].apply(lambda x: self.get_similar_item(x, filter_ctm=filter_ctm)).tolist()
        rec = self._extend_rec_popular(rec, N=N)

        return rec
    
    def _extend_rec_popular(self, rec, N):        
        if len(rec) < N:
            rec.extend(self.overall_top_purchases[:N])
            rec = rec[:N]
        return rec
        
        
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        rec = []
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)[1:]

        for user_score in similar_users:
			best_recs = self.own_recommender.recommend(user_score[0], csr_matrix(self.user_item_matrix).tocsr(), N=3)
			for one_rec in best_recs:
				if one_rec[0] in rec:
					continue
				rec.append(one_rec[0])
				break
            
        rec = self._extend_rec_popular(rec, N=N)
        
        return rec