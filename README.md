# Задание 2
Вначале преобразуем данные нам датафреймы так, чтобы они стали более пригодны к анализу и предсказанию:
```
import logging
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LinearRegression

orders_df = pd.read_csv("orders.csv")
partners_delays_df = pd.read_csv("partners_delays.csv")
partners_delays_df = partners_delays_df.rename(columns={"delivery_area_id": "delivery_area_id", "dttm": "date", "partners_cnt": "partners_cnt", "delay_rate": "delay_rate"})
new_df = pd.merge(orders_df, partners_delays_df,  how='inner', on=['delivery_area_id','date'])
cal = calendar()
dr = pd.date_range(start='2021-04-01', end='2021-11-30')
holidays = cal.holidays(start=dr.min(), end=dr.max())

new_df['Holiday'] = new_df['date'].isin(holidays)
new_df["date"] = pd.to_datetime(new_df["date"])
new_df['datehour'] = new_df['date'].dt.hour
new_df['time'] = [d.time() for d in new_df['date']]

new_df = new_df[["delivery_area_id", "date", 'Holiday', 'datehour', "orders_cnt", "partners_cnt", "delay_rate"]]

new_df["date"] = pd.to_datetime(new_df["date"])
new_df['day_of_week'] = new_df['date'].dt.day_name()
new_df = new_df[['delivery_area_id', 'date', 'day_of_week','Holiday', 'datehour', 'orders_cnt', 'partners_cnt', 'delay_rate']]

predictions = 3
# приводим dataframe к нужному формату
df = new_df
df = df.where(df['delivery_area_id']== 1)
df = df.where(df['day_of_week'] == "Monday")
df = df.where(df['datehour'] == 9)
df = df.where(df['Holiday'] == False)
df = df.drop(columns= ['day_of_week', 'datehour', 'partners_cnt', 'delay_rate'], axis = 1)
df = df.drop(columns= ['delivery_area_id'], axis = 1)
df = df.rename(columns= {"date": "ds", "orders_cnt": "y"})
df = df.dropna()
# отрезаем из обучающей выборки последние 30 точек, чтобы измерить на них качество
train_df = df[:-predictions] 
df.head(100)
df.isna().value_counts()

new_df.dtypes
new_df['datehour'] = new_df['date'].dt.hour

cal = calendar()
dr = pd.date_range(start='2021-04-01', end='2021-11-30')
holidays = cal.holidays(start=dr.min(), end=dr.max())

new_df['Holiday'] = new_df['date'].isin(holidays)*1
```
Результат:

![hackathon_1.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_1.png)

Далее вычисляем коэффициент `cnt`, который высчитывается как: `partners_cnt` / `orders_cnt` для каждой записи и удалим строки, где `delay_rate == 0`:

```
new_df['cnt'] = new_df['partners_cnt']/new_df['orders_cnt']
df_copy = df_copy[df_copy['delivery_area_id'] == 0]
```
Результат:

![hackathon_2.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_2.png)

После этого построим несколько графиков, из которых увидим, что распределение зависимости коэффициента `cnt` и уровня `delay_rate` примерно одинаково во всех случаях. Пример кода построения графика:
```
fig, ax = plt.subplots(figsize=(20, 8))
a = np.array([0.05] * 7)
ax.scatter(df_copy['cnt'].values, df_copy['delay_rate'].values, marker=None, cmap=None, vmin=None, alpha=None, linewidths=None, edgecolors='Green')
ax.plot(a)
ax.set_xlabel("заказы/партнеры", fontsize=14, labelpad=30)
ax.set_ylabel("delay_rate", fontsize=14, labelpad=30)
```
Построенные графики:
![hackathon_3.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_3.png)
![hackathon_4.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_4.png)
![hackathon_5.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_5.png)

Далее вычисляем коэффициент `coeff`, умножение на который позволит нам найти необходимое количество курьеров для каждой зоны для каждого дня:
```
delivery_area_id = new_df['delivery_area_id'].tolist()
datehour = new_df['datehour'].tolist()
cnt = new_df['cnt'].tolist()
delay_rate = new_df['delay_rate'].tolist()

d = {}

for i in range(0, len(cnt)):
    if delivery_area_id[i] not in d.keys():
        d[delivery_area_id[i]] = [[delay_rate[i]],[cnt[i]]]
    else:
        d[delivery_area_id[i]][0].append(delay_rate[i])
        d[delivery_area_id[i]][1].append(cnt[i])
       
result = []
c = 0
for i in range(593):
    for j in sorted(d[i][1]):
        for k in range(len(d[i][0])):
            if d[i][1][k] > j and d[i][0][k] > 0.05:
                c+=1
        if c / len(d[i][0]) < 0.1:
                result.append(j)
                break
        c = 0
    
result = list(result)

result = pd.DataFrame(result, columns={"coeff"})
```
Результат:

![hackathon_6.png](https://github.com/MOOsipenko/CMF-Hackathon/blob/main/hackathon_6.png)
