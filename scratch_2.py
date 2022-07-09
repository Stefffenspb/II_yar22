#Библиотеки
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


#Подготовим данные


def data_prepare(df):
    # нет доп описания компактного проживания по национальности
    df = df.drop(columns='Национальность')
    # вычислим количество часов сна
    df['Difference'] = df['Время пробуждения'].astype('datetime64[ns]') - df['Время засыпания'].astype('datetime64[ns]')
    df['Difference'] = df.Difference.dt.components['hours']
    #приберемся после вычисления
    df = df.drop(columns='Время засыпания')
    df = df.drop(columns='Время пробуждения')
    #В данных нет Перекоса
    #df = df.drop(columns='Этнос')
    df = df.drop(columns='Религия')
    #df = df.drop(columns='Профессия')


    df = df.drop(columns='Онкология')
    df = df.drop(columns='ВИЧ/СПИД')
    df = df.drop(columns='Туберкулез легких ')
    df = df.drop(columns='Образование')
    #Преобразуем справочники
    #df = df.replace({'Национальность': {'Русские': '1', 'Татары': '2','Таджики': '3','Немцы': '4','Евреи': '5','Азербайджанцы':'6','Другие национальности':'7','Чуваши':'8','Армяне':'9',
    #                                    'Украинцы':'12','Эстонцы':'13','Молдаване':'14','Мордва':'15','Киргизы':'16','Казахи':'17','Белорусы':'18','Башкиры':'19','Буряты':'20',
    #                                    'Удмурты':'21','Лезгины':'22'}})
    df = df.replace({'Профессия': {'дипломированные специалисты': '1',
                                   'квалифицированные работники сельского хозяйства и рыболовного': '2',
                                   'низкоквалифицированные работники': '3',
                                   'ремесленники и представители других отраслей промышленности': '4',
                                   'работники,  занятые в сфере обслуживания, торговые работники магазинов и рынков': '5',
                                   'ведение домашнего хозяйства': '6',
                                   'представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры': '7',
                                   'служащие': '8', 'операторы и монтажники установок и машинного оборудования': '9',
                                   'техники и младшие специалисты': '10', 'вооруженные силы': '11'}})
    df = df.replace({'Этнос': {'европейская': '1',
                               'прочее (любая иная этно-расовая группа, не представленная выше)': '2',
                               'другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)': '3'}})
    df = df.replace({'Пол': {'М': '1', 'Ж': '0'}})
    #df = df.replace({'Религия': {'Христианство': '1', 'Атеист / агностик': '2', 'Нет': '0', 'Ислам': '3', 'Другое': '4',
    #                             'Индуизм': '5'}})
    #df = df.replace({'Образование': {'5 - ВУЗ': '5', '4 - профессиональное училище': '4',
    #                                 '3 - средняя школа / закон.среднее / выше среднего': '3',
    #                                 '2 - начальная школа': '2'}})
    df = df.replace({'Семья': {'в браке в настоящее время': '0', 'в разводе': '1',
                               'гражданский брак / проживание с партнером': '2', 'вдовец / вдова': '3',
                               'никогда не был(а) в браке': '4',
                               'раздельное проживание (официально не разведены)': '5'}})
    df = df.replace(
        {'Статус Курения': {'Курит': '3', 'Никогда не курил(а)': '1', 'Бросил(а)': '2', 'Никогда не курил': '1'}})
    df = df.replace({'Частота пасс кур': {'1-2 раза в неделю': '1', '3-6 раз в неделю': '2',
                                          'не менее 1 раза в день': '3', '4 и более раз в день': '4',
                                          '2-3 раза в день': '5'}})
    df = df.replace(
        {'Алкоголь': {'употребляю в настоящее время': '3', 'никогда не употреблял': '1', 'ранее употреблял': '2'}})
    df['Difference'] = df['Difference'].mask(df['Difference'] < 7, 0)

    df['Difference'] = df['Difference'].mask((df['Difference'] >= 7) & (df['Difference'] <= 9), 1)
    df['Difference'] = df['Difference'].mask(df['Difference'] > 9, 2)
    df['Возраст алког'] = df['Возраст алког'].mask(df['Возраст алког'] > 3, 1)
    df['Сигарет в день'] = df['Сигарет в день'].mask(df['Сигарет в день'] < 7, 1)
    df['Сигарет в день'] = df['Сигарет в день'].mask((df['Сигарет в день'] >= 7) & (df['Difference'] < 20), 2)
    df['Сигарет в день'] = df['Сигарет в день'].mask(df['Сигарет в день'] >= 20, 3)

    df['Возраст курения'] = df['Возраст курения'].mask(df['Возраст курения'] < 10, 1)
    df['Возраст курения'] = df['Возраст курения'].mask((df['Возраст курения'] >= 10) & (df['Difference'] < 20), 2)
    df['Возраст курения'] = df['Возраст курения'].mask(df['Возраст курения'] >= 20, 3)


    df['Пол'] = df['Пол'].fillna('0')

    df['Частота пасс кур'] = df['Частота пасс кур'].fillna('0')
    df['Возраст алког'] = df['Возраст алког'].fillna('0')
    df['Возраст курения'] = df['Возраст курения'].fillna('0')
    df['Сигарет в день'] = df['Сигарет в день'].fillna('0')


    return df

#Обучение и Прогнозирование
def science(col, preds, filecsv_from, filecsv_to,colim):

    df_train = df
    for diag in colim:

        df_train.drop(columns=diag)

    X = df_train.drop(col, axis=1)
    y = df_train[col]
    test_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size, random_state=seed)
    num_folds = 10
    n_estimators = 100
    scoring = 'recall'
    models = []
    #Список Моделей

    models.append(('NB', GaussianNB()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('MLP', MLPClassifier()))
    models.append(('BG', BaggingClassifier(n_estimators=n_estimators)))
    models.append(('RF', RandomForestClassifier(n_estimators=n_estimators)))
    models.append(('ET', ExtraTreesClassifier(n_estimators=n_estimators)))
    models.append(('AB', AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME')))
    models.append(('GB', GradientBoostingClassifier(n_estimators=n_estimators)))




    # оцениваем модель на каждой итерации
    scores = []
    names = []
    results = []
    predictions = []
    msg_row = []
    best = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        names.append(name)
        results.append(cv_results)
        m_fit = model.fit(X_train, Y_train)
        m_predict = model.predict(X_validation)
        predictions.append(m_predict)
        m_score = model.score(X_validation, Y_validation)
        scores.append(m_score)
        msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(), cv_results.std(), m_score)
        msg_row.append(msg)
        print(msg)
        best.append(cv_results.mean())
        modi = model.fit(X, y)
        predi = modi.predict(X)
        predi = pd.DataFrame({col: predi})
        recall = recall_score(y, predi)
        print('recall_score', recall)

    index = best.index(max(best))
    #Выберем Лучшую
    best_train = models[index][1]

    df_test = pd.read_csv('data/test_dataset_test.csv')
    df_test.head()
    df_test_2 = pd.read_csv(filecsv_to)
    df_test_2.head()

    df_test_2 = df_test_2.rename(columns={'ID': 'ID_y'})
    df_merged = pd.merge(df_test, df_test_2, left_on='ID', right_on='ID_y', how='outer')
    df_merged = data_prepare(df_merged)
    df_merged = df_merged.drop(columns='ID_y')
    df_merged_id = pd.read_csv(filecsv_to)
    df_merged_id.head()

    df_merged = df_merged.drop(columns=col)
    df_merged_d = df_merged.drop(columns='ID')
    for diag in colim:
        df_merged_d.drop(columns=diag)
    gnb = best_train

    model = gnb.fit(X,  y)

    preds = model.predict(df_merged_d)

    preds = pd.DataFrame({col: preds})

    df_merged_id = df_merged_id.drop(columns=col)
    df_merged_id = pd.concat([df_merged_id, preds], axis=1)

    df_merged_id = df_merged_id[['ID', 'Артериальная гипертензия', 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда', 'Сердечная недостаточность',
              'Прочие заболевания сердца']]
    df_merged_id.to_csv(filecsv_to, index=False)
    #Значения предсказания

    return preds


df = pd.read_csv('data/train.csv')
df.head()

df = data_prepare(df)
df = df.drop(columns='ID_y')
df = df.drop(columns='ID')
#Соотношение Столбцов
print(df.skew())
#Построим Матрицу
# scatter_matrix(df)
# plt.show()
preds = []

filecsv_from = 'data/sample_solution.csv'
df_result = pd.read_csv(filecsv_from)
df_result.head()
df_merged_id = df_result['ID']
filecsv_to = 'result/sample_solution.csv'
df_result.to_csv(filecsv_to, index=False)

column_list = pd.Series(['Артериальная гипертензия', 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда', 'Сердечная недостаточность',
               'Прочие заболевания сердца'])
colim = column_list
colim.index=column_list
for col in column_list:
    colim = colim.drop(col)

    science(col, preds, filecsv_from, filecsv_to,colim)
#for col in column_list:
#    science(col, preds, filecsv_from, filecsv_to)








