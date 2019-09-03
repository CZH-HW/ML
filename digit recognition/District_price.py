import os 
import tarfile
from six.moves import urllib      #six.moves兼容Python2和Python3
import pandas as pd
import numpy as np
import hashlib


#################################################################################################################
#引入数据并划分训练集和验证集

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tg

# 获取数据函数，在工作空间创建目录，下载数据文件并解压
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)  #下载数据并指定保存路径
    housing_tgz = tarfile.open(tgz_path)   #打开压缩文件
    housing_tgz.extractall(path=housing_path)   #将压缩文件中的所有文件取出置于指定路径，默认为当前工作环境
    housing_tgz.close()


# 使用Pandas加载数据
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")  #整合csv文件的路径
    return pd.read_csv(csv_path)



# 创建测试集,随机取样
def split_train_test(data, test_ratio):
    '''
    这个方法可行，但是如果再次运行程序，就会产生一个不同的测试集！多次运行之后，就会得到整个数据集，这是需要避免的
    解决的办法之一是保存第一次运行得到的测试集，并在随后的过程加载
    另一种方法是在调用np.random.permutation()之前，设置随机数生成器的种子（比如np.random.seed(42)），以产生总是相同的洗牌指数（shuffled_indices)
    但是如果获取更新后的数据集，这两个方法都会失效
    '''
    test_set_size = int(len(data) * test_ratio)  #测试集的样本数 = 测试集占总样本比例 * 总样本数
    shuffled_indices = np.random.permutation(len(data)) #获取一个总样本数长度的随机排列的序列,len()获取轴0的长度，
    test_indices = shuffled_indices[:test_set_size]  #获取测试集在总样本的索引
    train_indices = shuffled_indices[test_set_size:] #获取训练集在总样本的索引
    return data.iloc[train_indices], data.iloc[test_indices]  #iloc基于索引位置来选取数据集



# 创建测试集V2,每次取样相同
'''
一个通常的解决办法是使用每个实例的识别码，以判定是否这个实例是否应该放入测试集（假设实例有单一且不变的识别码）。
例如，你可以计算出每个实例识别码的哈希值，只保留其最后一个字节，如果值小于等于51（约为256的20%），就将其放入测试集。
这样可以保证在多次运行中，测试集保持不变，即使更新了数据集。新的测试集会包含新实例中的20%，但不会有之前位于训练集的实例.
'''
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]      #id_column为识别码一列
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))   #获取测试集索引
    return data.loc[~in_test_set], data.loc[in_test_set]    #loc基于索引位置来选取数据集


# Scikit-Learn提供了一些函数，可以用多种方式将数据集分割成多个子集
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)    



# 分层采样
'''
不能有过多的分层，且每个分层都要足够大。后面的代码通过将收入中位数除以 1.5（以限制收入分类的数量）
创建了一个收入类别属性，用ceil对值舍入（以产生离散的分类），然后将所有大于 5 的分类归入到分类 5
income_cat属性用作临时存储分层数据的列
'''
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)      #ceil()将数据中的元素替换成大于相应元素的最小整数
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)     #where将income_cat列大于5的归类到5

#根据收入分类，进行分层采样，使用Scikit-Learn的StratifiedShuffleSplit类(交叉验证)
from sklearn.model_selection import StratifiedShuffleSplit
#实例化，参数n_splits是将训练数据分成train,test对的组数，可根据需要进行设置，默认为10
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)   

for train_index, test_index in split.split(housing, housing["income_cat"]):    #split()为生成器
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 删除训练集和验证集income_cat属性，使数据回到初始状态
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


########################################################################################################################


# 数据探索，数据可视化
# 用在训练算法之前，选择合适的特征
housing = strat_train_set.copy()  #创建一个副本
housing.plot(kind="scatter", x="longitude", y="latitude")  #存在地理信息（纬度和经度），创建一个所有街区的散点图来数据可视化
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)   #alpha表示透明度，可视化密度分布

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,  
    s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, 
)   #每个圈的半径表示街区的人口（选项s），颜色代表价格（选项c），用预先定义的名为jet的颜色图（选项cmap），它的范围是从蓝色（低价）到红色（高价）
plt.legend()

# 查找关联
corr_matrix = housing.corr()  #corr()方法计算出每对属性间的标准相关系数
corr_matrix["median_house_value"].sort_values(ascending=False)  #查看每个属性和房价中位数的关联度

from pandas.tools.plotting import scatter_matrix    
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))   #scatter_matrix()能画出每个数值属性对每个其它数值属性的图

#最有希望用来预测房价中位数的属性是收入中位数，因此将这张图放大
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# 属性组合试验
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


##########################################################################################################################


#将训练集的特征和标签分开
housing = strat_train_set.drop("median_house_value", axis=1)   #当inplace参数为默认值False时drop()创建了一份数据的备份，而不影响strat_train_set
housing_labels = strat_train_set["median_house_value"].copy()  #获取标签

#数据清洗，处理数据缺失等
housing.dropna(subset=["total_bedrooms"])    #去掉对应的街区（行）
housing.drop("total_bedrooms", axis=1)       #去掉整个属性

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)     #将缺失值进行赋值（中位数）

#Scikit-Learn 提供了一个方便的类来处理缺失值：Imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)  #只有数值属性才能算出中位数，创建一份不包括文本属性ocean_proximity的数据副本
imputer.fit(housing_num)    #imputer计算出了每个属性的中位数，并将结果保存在了实例变量statistics_中
X = imputer.transform(housing_num)    #对训练集进行转换，通过将缺失值替换为中位数
housing_tr = pd.DataFrame(X, columns=housing_num.columns)    #numpy数组转化魏Pandas的DataFrame结构

#处理文本和类别属性
from sklearn.preprocessing import LabelEncoder    #转换器LabelEncoder
from sklearn.preprocessing import OneHotEncoder    #编码器OneHotEncoder，用于将整书分类值转变为独热向量
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)   #从文本分类到整数分类，返回一个一维数组
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))   #-1参数使得数组根据另一个参数1转化为列向量2D，返回一个SciPy稀疏矩阵，而不是NumPy数组

#使用类LabelBinarizer，我们可以用一步执行上述这两个转换（从文本分类到整数分类，再从整数分类到独热向量）
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)   #默认返回一个numpy数组，向构造器LabelBinarizer传递sparse_output=True，就可以得到一个稀疏矩阵


#自定义转换器，特征之间选择和组合
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):    # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room     #超参数add_bedrooms_per_room用来选择是否添加这个属性
    def fit(self, X, y=None):
        return self    # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#特征缩放
#线性函数归一化（Min-Max scaling）和标准化（standardization）

############################################################################################################################


#转换流水线
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])





############################################################################################################################

#在训练集上训练和评估
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#使用 Scikit-Learn 的mean_squared_error函数，用全部训练集来计算下这个回归模型的 RMSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)


#使用交叉验证做更佳的评估
from sklearn.tree import DecisionTreeRegressor      #决策树
from sklearn.model_selection import cross_val_score
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
#K 折交叉验证（K-fold cross-validation）功能期望的是效用函数（越大越好）而不是损失函数（越低越好），因此得分函数实际上与 MSE 相反（即负值）
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)   
rmse_scores = np.sqrt(-scores)

#模型微调    
from sklearn.model_selection import GridSearchCV   #网格搜索
from sklearn.ensemble import RandomForestRegressor   #随机森林
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)