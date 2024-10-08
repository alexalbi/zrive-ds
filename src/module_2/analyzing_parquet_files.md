```python
import pandas as pd 
import os
```


```python
folder = 'data/groceries/sampled-datasets'

parquet_files = [f for f in os.listdir(folder)]
for f in parquet_files:
    locals()[f'df_{f.rsplit(".",1)[0]}'] = pd.read_parquet(str(folder+'/'+f))
     
```


```python
df_orders.info()
df_orders.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB
    

Missing values:
  - Users: we have nulls in the ‘user_nuts1’ columns and the ‘count_people’ columns and derivatives, which are attributed to lack of information added by users. (4983,10)


Hypotheses about what we think we know:
1. London and South East UK areas have a higher level of income, assuming this fact they should spend more than the rest of the areas. (To validate this we will use the 'user_nuts1' variable from the user dataframe).
2. Users segmented as Top-Up should make small purchases with a high frequency and users segmented as Proposition should make large purchases infrequently. (To validate this we will use the 'user_segment' variable from the user dataframe).

What we would like to know:
1. Which products sell the most by geographic area, to see if there is variation in specific products between different areas.
2. Check if there is a relationship between product_type and cart abandonment.

### Hypothesis 1:
The price of orders is higher in regions with higher level of income


```python
orders_nuts = df_orders.merge(df_users[['user_id', 'user_nuts1']], on='user_id', how='left').explode('ordered_items')
df_merged = orders_nuts.merge(df_inventory[['variant_id', 'price']], left_on='ordered_items', right_on='variant_id', how='left')
df_merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 107958 entries, 0 to 107957
    Data columns (total 9 columns):
     #   Column          Non-Null Count   Dtype         
    ---  ------          --------------   -----         
     0   id              107958 non-null  int64         
     1   user_id         107958 non-null  object        
     2   created_at      107958 non-null  datetime64[us]
     3   order_date      107958 non-null  datetime64[us]
     4   user_order_seq  107958 non-null  int64         
     5   ordered_items   107958 non-null  object        
     6   user_nuts1      107148 non-null  object        
     7   variant_id      92361 non-null   float64       
     8   price           92361 non-null   float64       
    dtypes: datetime64[us](2), float64(2), int64(2), object(3)
    memory usage: 7.4+ MB
    

After the merge of the dataframes, we have 15597 null rows at 'variant_id' and 'price', the variables we have joined. We can assume that there are products that were purchased and are no longer in ‘inventory.parquet’ or that they have changed their ‘id’. But we have enough data to test the hypothesis and we will delete these rows.


```python
df_merged = df_merged.dropna(subset=['price'])
df_total_price = df_merged.groupby('id')['price'].sum().reset_index()
nuts_price = df_hyp1.merge(df_total_price, on='id', how='left')

summary_nuts_price = nuts_price.groupby('user_nuts1')['price'].describe()
print(summary_nuts_price)
```

                  count       mean        std    min      25%    50%    75%  \
    user_nuts1                                                                
    UKC          2680.0  62.128377  24.261948   1.99  47.0600  54.16  75.29   
    UKD          5870.0  56.727482  22.132205   2.98  45.9700  52.55  61.87   
    UKE          5864.0  61.849615  30.532447   4.79  45.5000  52.62  65.03   
    UKF          5781.0  57.506585  21.495969   4.76  45.5300  52.28  63.48   
    UKG          5527.0  58.055602  22.108602   5.58  46.8900  52.71  63.84   
    UKH         10863.0  66.013245  35.378781   5.28  46.9000  54.44  71.47   
    UKI         26381.0  66.440882  37.514547   0.99  47.7900  53.23  69.55   
    UKJ         18159.0  66.439914  35.881358   1.99  47.3000  55.06  73.24   
    UKK         13926.0  61.172730  27.130163   0.99  46.1200  53.95  66.26   
    UKL          4848.0  68.303267  35.767825  10.47  50.1925  56.94  72.77   
    UKM          6749.0  65.129874  37.954970   7.08  46.6500  53.94  70.47   
    UKN            53.0  53.518679  20.944562   7.38  44.2700  56.26  76.75   
    
                   max  
    user_nuts1          
    UKC         152.39  
    UKD         191.88  
    UKE         200.77  
    UKF         168.31  
    UKG         160.15  
    UKH         221.31  
    UKI         256.47  
    UKJ         319.80  
    UKK         231.33  
    UKL         271.89  
    UKM         319.80  
    UKN          76.75  
    

### Hypothesis 2:
Users segmented as Top-Up should make small purchases with a high frequency and users segmented as Proposition should make large purchases infrequently. (To validate this we will use the 'user_segment' variable from the user dataframe).


```python
df_hyp2 = df_hyp1.merge(df_users[['user_id', 'user_segment']], on='user_id', how='left')

summary_segment_price = df_hyp2.groupby('user_segment')['price'].describe()
print(summary_segment_price)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[29], line 3
          1 df_hyp2 = df_hyp1.merge(df_users[['user_id', 'user_segment']], on='user_id', how='left')
    ----> 3 summary_segment_price = df_hyp2.groupby('user_segment')['price'].describe()
          4 print(summary_segment_price)
    

    File c:\Users\ALEX\OneDrive\Escritorio\PROGRAMACION\ZRIVE\zrive-ds\.venv\Lib\site-packages\pandas\core\groupby\generic.py:1771, in DataFrameGroupBy.__getitem__(self, key)
       1764 if isinstance(key, tuple) and len(key) > 1:
       1765     # if len == 1, then it becomes a SeriesGroupBy and this is actually
       1766     # valid syntax, so don't raise
       1767     raise ValueError(
       1768         "Cannot subset columns with a tuple with more than one element. "
       1769         "Use a list instead."
       1770     )
    -> 1771 return super().__getitem__(key)
    

    File c:\Users\ALEX\OneDrive\Escritorio\PROGRAMACION\ZRIVE\zrive-ds\.venv\Lib\site-packages\pandas\core\base.py:244, in SelectionMixin.__getitem__(self, key)
        242 else:
        243     if key not in self.obj:
    --> 244         raise KeyError(f"Column not found: {key}")
        245     ndim = self.obj[key].ndim
        246     return self._gotitem(key, ndim=ndim)
    

    KeyError: 'Column not found: price'


### Question 1:
Which products sell the most by geographic area, to see if there is variation in specific products between different areas.


```python
df_q1 = df_orders.merge(df_users[['user_id', 'user_nuts1']], on='user_id', how='left') 
df_q1_expanded = df_q1.explode('ordered_items')
df_q1_merged = df_q1_expanded.merge(df_inventory[['variant_id', 'product_type']], left_on='ordered_items', right_on='variant_id', how='left')
df_q1_merged = df_q1_merged.dropna(subset=['product_type'])
df_q1_merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 92361 entries, 70 to 107957
    Data columns (total 9 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              92361 non-null  int64         
     1   user_id         92361 non-null  object        
     2   created_at      92361 non-null  datetime64[us]
     3   order_date      92361 non-null  datetime64[us]
     4   user_order_seq  92361 non-null  int64         
     5   ordered_items   92361 non-null  object        
     6   user_nuts1      91716 non-null  object        
     7   variant_id      92361 non-null  float64       
     8   product_type    92361 non-null  object        
    dtypes: datetime64[us](2), float64(1), int64(2), object(4)
    memory usage: 7.0+ MB
    


```python
products_by_zone  = df_q1_merged.groupby(['user_nuts1', 'product_type']).size().unstack(fill_value=0)
relative_products_by_zone = products_by_zone.div(products_by_zone.sum(axis=1), axis=0)
```


```python
# Obtener los 3 productos más comunes por zona
top_3_products_per_zone = relative_products_by_zone.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)
print(top_3_products_per_zone)
```

    user_nuts1
    UKC    [cleaning-products, tins-packaged-foods, toile...
    UKD    [cleaning-products, tins-packaged-foods, toile...
    UKE    [cleaning-products, tins-packaged-foods, dishw...
    UKF    [cleaning-products, tins-packaged-foods, toile...
    UKG    [cleaning-products, tins-packaged-foods, toile...
    UKH    [cleaning-products, tins-packaged-foods, toile...
    UKI    [cleaning-products, tins-packaged-foods, toile...
    UKJ    [tins-packaged-foods, cleaning-products, long-...
    UKK    [tins-packaged-foods, long-life-milk-substitut...
    UKL    [tins-packaged-foods, cleaning-products, long-...
    UKM    [long-life-milk-substitutes, cleaning-products...
    UKN    [cleaning-products, delicates-stain-remover, d...
    dtype: object
    

### Question 2:
Check if there is a relationship between product_type and cart abandonment.


```python
products_sold_by_type = df_orders.explode('ordered_items').merge(df_inventory[['variant_id',
                                                                               'product_type']], left_on='ordered_items', 
                                                                 right_on='variant_id', how='left')['product_type'].value_counts().reset_index()
products_sold_by_type.columns = ['product_type', 'sold']
```


```python
products_abandoned_by_type = df_abandoned_carts.explode('variant_id').merge(df_inventory[['variant_id', 
                                                                     'product_type']], on='variant_id', 
                                                       how='left')['product_type'].value_counts().reset_index()
products_abandoned_by_type.columns = ['product_type', 'abandoned']
```


```python
sold_vs_abandoned = products_sold_by_type.merge(products_abandoned_by_type, on='product_type', how='left')
sold_vs_abandoned['abandonment_rate'] = sold_vs_abandoned['abandoned'] / (sold_vs_abandoned['abandoned'] + sold_vs_abandoned['sold'])
```


```python
df_q2.sort_values('abandonment_rate', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>sold</th>
      <th>abandoned</th>
      <th>abandonment_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57</th>
      <td>mixed-bundles</td>
      <td>2</td>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>medicine-treatments</td>
      <td>27</td>
      <td>30</td>
      <td>0.526316</td>
    </tr>
    <tr>
      <th>46</th>
      <td>household-sundries</td>
      <td>146</td>
      <td>116</td>
      <td>0.442748</td>
    </tr>
    <tr>
      <th>47</th>
      <td>medicines-treatments</td>
      <td>97</td>
      <td>72</td>
      <td>0.426036</td>
    </tr>
    <tr>
      <th>45</th>
      <td>superfoods-supplements</td>
      <td>159</td>
      <td>107</td>
      <td>0.402256</td>
    </tr>
    <tr>
      <th>50</th>
      <td>low-no-alcohol</td>
      <td>58</td>
      <td>39</td>
      <td>0.402062</td>
    </tr>
    <tr>
      <th>36</th>
      <td>spirits-liqueurs</td>
      <td>578</td>
      <td>350</td>
      <td>0.377155</td>
    </tr>
    <tr>
      <th>43</th>
      <td>cider</td>
      <td>215</td>
      <td>114</td>
      <td>0.346505</td>
    </tr>
    <tr>
      <th>24</th>
      <td>washing-capsules</td>
      <td>1127</td>
      <td>581</td>
      <td>0.340164</td>
    </tr>
    <tr>
      <th>52</th>
      <td>maternity</td>
      <td>40</td>
      <td>20</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_q2.sort_values('abandonment_rate', ascending=False).tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>sold</th>
      <th>abandoned</th>
      <th>abandonment_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>tea</td>
      <td>1232</td>
      <td>315</td>
      <td>0.203620</td>
    </tr>
    <tr>
      <th>40</th>
      <td>deodorant</td>
      <td>299</td>
      <td>75</td>
      <td>0.200535</td>
    </tr>
    <tr>
      <th>33</th>
      <td>coffee</td>
      <td>729</td>
      <td>175</td>
      <td>0.193584</td>
    </tr>
    <tr>
      <th>37</th>
      <td>baby-toddler-food</td>
      <td>484</td>
      <td>116</td>
      <td>0.193333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cereal</td>
      <td>3014</td>
      <td>700</td>
      <td>0.188476</td>
    </tr>
    <tr>
      <th>42</th>
      <td>pet-care</td>
      <td>238</td>
      <td>52</td>
      <td>0.179310</td>
    </tr>
    <tr>
      <th>25</th>
      <td>bin-bags</td>
      <td>1083</td>
      <td>232</td>
      <td>0.176426</td>
    </tr>
    <tr>
      <th>53</th>
      <td>adult-incontinence</td>
      <td>39</td>
      <td>7</td>
      <td>0.152174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>long-life-milk-substitutes</td>
      <td>6637</td>
      <td>1134</td>
      <td>0.145927</td>
    </tr>
    <tr>
      <th>32</th>
      <td>baby-milk-formula</td>
      <td>787</td>
      <td>102</td>
      <td>0.114736</td>
    </tr>
  </tbody>
</table>
</div>



### Insights:
- It confirms the hypothesis that London and South East UK, being the areas with higher income levels, have a higher expenditure on purchases, although there are areas with a similar purchase price that we did not expect, such as Wales.
- The hypothesis about user segment is confirmed, as the Top Up consumer makes more purchases than the Proposition consumer, and the Proposition consumer makes purchases with higher spending.

- There's no clear reason to say that de area influences on the type of the products are bought. The most sold products are similar in all areas.
- The type of product clearly influences the likelihood of it being abandoned in the cart. Products considered more ‘necessary’ or essential, such as baby food or bin bags, have a low abandonment rate, menawhile products that require more consideration, such as medicines, alcoholic drinks, are more likely to be abandoned.
