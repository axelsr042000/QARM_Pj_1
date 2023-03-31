"""
LOAD PACKAGES
"""
import numpy as np
import pandas as pd
import scipy.optimize as sc
from scipy.stats import norm, t
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


########################################################################################################################
"""
PATH TO THE FILE
"""
path_to_file = '/home/itopie/Bureau/HEC Lausanne/QARM/QARM_Pj_1/Data/data.xlsx'
########################################################################################################################
"""
OPEN DATA
"""


def createTable(SheetName, NameTable, Deleted_col):
    NameTable = pd.read_excel(path_to_file, header=None, sheet_name=SheetName, engine='openpyxl')
    # Transpose the table
    NameTable = NameTable.transpose()
    # Select columns that do not contain "#ERROR#" in the first row
    mask = NameTable.iloc[0, :] != '#ERROR'
    NameTable = NameTable.loc[:, mask]
    # Put dates in place of the index column
    NameTable.set_index(NameTable.iloc[:, 0], inplace=True)  # set Date column as the index column
    # Delete 1st row of dates
    NameTable = NameTable.drop(columns=0)
    # Rename column names
    NameTable.columns = NameTable.iloc[0, :]
    # Delete first row
    NameTable = NameTable.iloc[1:]
    # Get current column names
    current_cols = list(NameTable.columns)
    # Rename columns by taking out what is after the -
    new_cols = [col.split(" -")[0] for col in current_cols]
    NameTable.columns = new_cols
    # Replace all NaT by NaN
    NameTable = NameTable.replace({pd.NaT: np.nan})
    # Convert colomn index into date format (could work without for some computers, probably tahnks to their excel
    # version, but being cautious we add it)
    NameTable.index = pd.to_datetime(NameTable.index)
    # Delete column with not enough data (such as GRAFTECH), because lead
    NameTable = NameTable.drop(Deleted_col, axis=1)

    return NameTable


# Create table for the differents parameters
df_MV = createTable('MV_DAILY', 'data_MV', 'GRAFTECH INTERNATIONAL')
df_TOT_RET = createTable('TOTAL_RETURN_INDEX', 'data_TOTAL_RETURN', 'GRAFTECH INTERNATIONAL')
df_REVENUES = createTable('REVENUES(MONTHLY)', 'data_REVENUES',
                          'GRAFTECH INTERNTL')  # Has not the same col names as others, but it is changed after.
df_CO2 = createTable('CO2 MONTHLY', 'data_CO2', 'GRAFTECH INTERNATIONAL')

df_company_info = pd.read_excel(path_to_file, header=0, sheet_name='Group_A', engine='openpyxl')
# Delete company with error in values and GRAFTECH (134) to keep consistent with other tables
df_company_info = df_company_info.drop([1, 32, 96, 132, 134, 138])

# For Daily table, transform them into monthly table
df_MV = df_MV.resample('M').last()
df_TOT_RET = df_TOT_RET.resample('M').last()

# For initial monthly data (revenues & CO2), put the same day (last day of month)
df_REVENUES.set_index(df_MV.index, inplace=True)  # set Date column as the index column
df_CO2.set_index(df_MV.index, inplace=True)  # set Date column as the index column

# TOT_RET does not have same name column as other, so put same name
df_REVENUES.columns = df_TOT_RET.columns
########################################################################################################################
"""
SiMPLE RETURNS
"""


def simple_returns(r0, r1):
    if r0 > 0:
        returns = ((r1 - r0) / r0)
    else:
        returns = np.nan

    return returns


# Create a new DataFrame to store the simple returns
df_returns = df_TOT_RET.copy()
# Put NaN in first row since returns begin in 2nd row
df_returns.iloc[0, :] = np.nan


# Put same column name as in df_TOT_RET
for col_name in df_TOT_RET.columns:
    for i in range(1, len(df_TOT_RET)):
        x0 = df_TOT_RET[col_name][i - 1]
        x1 = df_TOT_RET[col_name][i]
        df_returns[col_name][i] = simple_returns(x0, x1)


# """DROP ASSETS WITH LESS THAN 6 YEARS OF RETURNS IN COMMUN WITH OTHER ASSETS"""
# # Nombre minimum de mois de rendements en commun
# min_months = 72
#
# # Liste pour stocker les noms des colonnes avec moins de 72 mois de rendements en commun
# columns_to_drop = []
#
# # Boucle sur les colonnes du dataframe
# for i, col1 in enumerate(df_returns.columns):
#     # Compte le nombre de mois de rendements non-nuls pour la colonne i
#     months_i = df_returns[col1].count()
#     # Continue si la colonne i a au moins min_months mois de rendements non-nuls
#     if months_i >= min_months:
#         # Liste des dates pour lesquelles la colonne i a des rendements non-nuls
#         dates_i = df_returns.index[df_returns[col1].notnull()].tolist()
#         # Boucle sur les colonnes suivantes du dataframe
#         for col2 in df_returns.columns[i+1:]:
#             # Compte le nombre de mois de rendements non-nuls pour la colonne j
#             months_j = df_returns[col2].count()
#             # Continue si la colonne j a au moins min_months mois de rendements non-nuls
#             if months_j >= min_months:
#                 # Liste des dates pour lesquelles la colonne j a des rendements non-nuls
#                 dates_j = df_returns.index[df_returns[col2].notnull()].tolist()
#                 # Nombre de mois de rendements en commun
#                 months_common = len(set(dates_i).intersection(set(dates_j)))
#                 # Si le nombre de mois de rendements en commun est inférieur à min_months,
#                 # ajouter les noms des colonnes à la liste columns_to_drop
#                 if months_common < min_months:
#                     columns_to_drop.append(col1)
#                     columns_to_drop.append(col2)
#
# # Supprime les doublons de la liste des colonnes à dropper
# columns_to_drop = list(set(columns_to_drop))

# Affiche la liste des colonnes à dropper
# print("Colonnes à dropper : ", columns_to_drop)
# df_returns.drop(columns_to_drop, axis=1, inplace=True)

df_returns = df_returns.iloc[1:]
df_returns = df_returns.dropna(axis=1)
mean_returns = df_returns.mean()
cov_matrix = df_returns.cov()


# TEST COV_MATRIX
# determinant = print(np.linalg.det(cov_matrix))
# eig_values, eig_vectors = np.linalg.eig(cov_matrix)
# if np.all(eig_values > 0):
#     print("La matrice est définie positive.")
# else:
#     print("La matrice n'est pas définie positive.")
# corr_matrix = df_returns.corr()


# # tracer le graphique pour chaque actif
# for column in df_returns.columns:
#     plt.plot(df_returns.index, df_returns[column], label=column)
#
# # ajouter les titres et légendes
# plt.title('Rendements des actifs')
# plt.xlabel('Dates')
# plt.ylabel('Rendements')
# # plt.legend(loc='upper left')
#
# # afficher le graphique
# plt.show()
########################################################################################################################
"""
QUESTION 1.1

# In[1.1]:

# Create a table for all descriptive statistics of df_REVENUES

df_stats_REVENUES = pd.DataFrame()
df_stats_REVENUES['Min'] = df_REVENUES.min()
df_stats_REVENUES['Max'] = df_REVENUES.max()

# Create a table for all descriptive statistics of df_TOT_RET

df_stats_TOT_RET = pd.DataFrame()
df_stats_TOT_RET['Min'] = df_TOT_RET.min()
df_stats_TOT_RET['Max'] = df_TOT_RET.max()

# Create a table for all descriptive statistics of df_CO2

df_stats_CO2 = pd.DataFrame()
df_stats_CO2['Min'] = df_CO2.min()
df_stats_CO2['Max'] = df_CO2.max()

# Create a table for all descriptive statistics of df_returns

df_stats_returns = pd.DataFrame()
df_stats_returns['Mean'] = df_returns.mean()
df_stats_returns['Variance'] = df_returns.var(ddof=1)
df_stats_returns['Skewness'] = df_returns.skew()
df_stats_returns['Kurtosis'] = df_returns.kurtosis()
df_stats_returns['Min'] = df_returns.min()
df_stats_returns['Max'] = df_returns.max()

# Modifying the index of df_company_info into specific name for each corporation

df_company_info = df_company_info.set_index(df_stats_returns.index)

# create a dict to classify the companies name by their sector

groups = df_company_info.groupby('GICSSector', group_keys=False)
sector_dict = {name: group for name, group in groups}


# ere are the different sector :

# Communication Services
# Consumer Discretionary
# Consumer Staples
# Energy
# Financials
# Health Care
# Industrials
# Information Technology
# Materials
# Real Estate
# Utilities


CSe_df = sector_dict['Communication Services']
CD_df = sector_dict['Consumer Discretionary']
CSt_df = sector_dict['Consumer Staples']
E_df = sector_dict['Energy']
F_df = sector_dict['Financials']
HC_df = sector_dict['Health Care']
I_df = sector_dict['Industrials']
IT_df = sector_dict['Information Technology']
M_df = sector_dict['Materials']
RE_df = sector_dict['Real Estate']
U_df = sector_dict['Utilities']

# Creating all the dataframes, for each sector, containing the companies, and their return, specific for each sector

# transpose the matrix return to ba able to use it for the different sector
df_returns_trans = df_returns.transpose()

# Turn all df of each sectors into lists
CSe_list = CSe_df.index.tolist()
CD_list = CD_df.index.tolist()
CSt_list = CSt_df.index.tolist()
E_list = E_df.index.tolist()
F_list = F_df.index.tolist()
HC_list = HC_df.index.tolist()
I_list = I_df.index.tolist()
IT_list = IT_df.index.tolist()
M_list = M_df.index.tolist()
RE_list = RE_df.index.tolist()
U_list = U_df.index.tolist()

# Create the df containing the return, separated by sectors
CSe_return_df = df_returns_trans.loc[CSe_list].copy()
CD_return_df = df_returns_trans.loc[CD_list].copy()
CSt_return_df = df_returns_trans.loc[CSt_list].copy()
E_return_df = df_returns_trans.loc[E_list].copy()
F_return_df = df_returns_trans.loc[F_list].copy()
HC_return_df = df_returns_trans.loc[HC_list].copy()
I_return_df = df_returns_trans.loc[I_list].copy()
IT_return_df = df_returns_trans.loc[IT_list].copy()
M_return_df = df_returns_trans.loc[M_list].copy()
RE_return_df = df_returns_trans.loc[RE_list].copy()
U_return_df = df_returns_trans.loc[U_list].copy()

# We now want to plot the distributions of each sector's return
# We firstly turn the df into an array, with each return being the mean of the return from the companies of
# the same sectors

mean_CSe = CSe_return_df.mean()
mean_CD = CD_return_df.mean()
mean_CSt = CSt_return_df.mean()
mean_E = E_return_df.mean()
mean_F = F_return_df.mean()
mean_HC = HC_return_df.mean()
mean_I = I_return_df.mean()
mean_IT = IT_return_df.mean()
mean_M = M_return_df.mean()
mean_RE = RE_return_df.mean()
mean_U = U_return_df.mean()

# Create a kernel density estimate plot of the returns for each sector
sns.kdeplot(mean_CSe, label='mean_CSe')
sns.kdeplot(mean_CD, label='mean_CD')
sns.kdeplot(mean_CSt, label='mean_CSt')
sns.kdeplot(mean_E, label='mean_E')
sns.kdeplot(mean_F, label='mean_F')
sns.kdeplot(mean_HC, label='mean_HC')
sns.kdeplot(mean_I, label='mean_I')
sns.kdeplot(mean_IT, label='mean_IT')
sns.kdeplot(mean_M, label='mean_M')
sns.kdeplot(mean_RE, label='mean_RE')
sns.kdeplot(mean_U, label='mean_U')

# Set the x-axis label, y-axis label, and title
plt.xlabel('Return')
plt.ylabel('Density')
plt.title('Distribution of Returns')

# Add a legend
plt.legend()

# Show the plot
plt.show()
"""
########################################################################################################################
"""
QUESTION 1.2
EFFICIENT FRONTIER METHOD 1
"""


def weights_creator(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum()
    return rand


def portfolio_perf(w, m_returns, cov_mat):
    port_returns = np.sum(m_returns * w) * 12
    port_std = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(12)
    return port_returns, port_std


def negative_sharpe_ratio(w, m_returns, cov_mat, rf=0):
    port_returns, port_std = portfolio_perf(w, m_returns, cov_mat)
    return - (port_returns - rf) / port_std


def max_sharpe_ratio(m_returns, cov_mat, rf=0, constraint_set=(-1, 1)):
    """Min negative Sharpe Ratio by altering the weights of the portfolio"""
    num_assets = len(m_returns)
    args = (m_returns, cov_mat, rf)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for i in range(num_assets))
    res = sc.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                         method="SLSQP", bounds=bounds, constraints=constraints)
    return res


def portfolio_var(w, m_returns, cov_mat):
    return portfolio_perf(w, m_returns, cov_mat)[1]


def min_var(m_returns, cov_mat, constraint_set=(-1, 1)):
    """Min the portfolio variance by altering the weights/allocation of assets in the portfolio"""
    num_assets = len(m_returns)
    args = (m_returns, cov_mat)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for i in range(num_assets))
    res = sc.minimize(portfolio_var, num_assets * [1. / num_assets], args=args,
                         method="SLSQP", bounds=bounds, constraints=constraints)
    return res


# weights = weights_creator(returns)
#
# portfolio_returns, portfolio_std = portfolio_perf(weights, mean_returns, cov_matrix)
# print(round(portfolio_returns * 100, 2), round(portfolio_std * 100, 2))
#
# sharpe_ratio_result = max_sharpe_ratio(mean_returns, cov_matrix)
# result_max_sharpe_ratio, result_max_sharpe_ratio_weights = sharpe_ratio_result["fun"], sharpe_ratio_result["x"]
# print(result_max_sharpe_ratio, result_max_sharpe_ratio_weights)
#
# min_var_result = min_var(mean_returns, cov_matrix)
# min_var, max_var_weights = min_var_result["fun"], min_var_result["x"]
# print(min_var, max_var_weights)


def port_return(w, m_returns, cov_mat):
    return portfolio_perf(w, m_returns, cov_mat)[0]


def efficient_optimization(m_returns, cov_mat, return_target, constraint_set=(-1, 1)):
    """For each return, we want to optimise the portfolio for min variance"""
    num_assets = len(m_returns)
    args = (m_returns, cov_mat)

    constraints = ({"type": "eq", "fun": lambda x: port_return(x, m_returns, cov_mat) - return_target},
                   {"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for i in range(num_assets))
    eff_opt = sc.minimize(portfolio_var, num_assets * [1. / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return eff_opt


def calculated_results(m_returns, cov_mat, rf=0, constraint_set=(-1, 1)):
    """Read in mean, cov matrix, and other financial information
    Output, Max SR, Min Vol, efficient frontier"""
    # Max Sharpe Ratio Portfolio
    max_sr_portfolio = max_sharpe_ratio(m_returns, cov_mat)
    max_sr_returns, max_sr_std = portfolio_perf(max_sr_portfolio["x"], m_returns, cov_mat)
    max_sr_allocation = pd.DataFrame(max_sr_portfolio["x"], index=m_returns.index, columns=["Allocation"])
    max_sr_allocation.Allocation = [round(i * 100, 0) for i in max_sr_allocation.Allocation]

    # Min Volatility Portfolio
    min_vol_portfolio = min_var(m_returns, cov_mat)
    min_vol_returns, min_vol_std = portfolio_perf(min_vol_portfolio["x"], m_returns, cov_mat)
    min_vol_allocation = pd.DataFrame(min_vol_portfolio["x"], index=m_returns.index, columns=["Allocation"])
    min_vol_allocation.Allocation = [round(i * 100, 0) for i in min_vol_allocation.Allocation]

    # Efficient Frontier
    efficient_list = []
    efficient_weights = []
    target_return = np.linspace(min_vol_returns, max_sr_returns, 10)
    # target_return = np.linspace(0.075, 0.18, 3)
    for target in target_return:
        efficient_list.append(efficient_optimization(m_returns, cov_mat, target)["fun"])
        efficient_weights.append(efficient_optimization(m_returns, cov_mat, target)["x"])

    max_sr_returns, max_sr_std = round(max_sr_returns * 100, 2), round(max_sr_std * 100, 2)
    min_vol_returns, min_vol_std = round(min_vol_returns * 100, 2), round(min_vol_std * 100, 2)

    return max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return, efficient_weights


def ef_graph(option, m_returns, cov_mat, m_returns_3, std_3, m_returns_4, std_4,
             nbr_samples, rf=0, constraint_set=(-1, 1)):
    """Return a graph ploting the min vol, max SR and efficient frontier"""
    max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return, efficient_weights = calculated_results(m_returns, cov_mat, rf, constraint_set)

    if option == 2:

        # Max SR
        plt.scatter(max_sr_std, max_sr_returns, color='red', s=50, edgecolors='black')

        # Min Vol
        plt.scatter(min_vol_std, min_vol_returns, color='green', s=50, edgecolors='black')

        # Efficient Frontier
        plt.plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
                 color='black', linestyle='-.', linewidth=2)

        plt.plot(std_4[0, :], m_returns_4[0, :], color='purple', linestyle='-.', linewidth=2)

        for nbr in range(nbr_samples):
            plt.plot(std_3[nbr, :], m_returns_3[nbr, :],
                     color='blue', linestyle='-.', linewidth=0.5)

        plt.title('Efficient Frontiers')
        plt.xlabel('Annualized Volatility (%)')
        plt.ylabel('Annualized Return (%)')
        plt.legend(['Maximum Sharpe Ratio', 'Minimum Volatility',
                    'Historical Efficient Frontier', 'Resampled Efficient Frontier',
                    'Simulated Efficient Frontiers'], loc='best')

    else:

        # Max SR
        plt.scatter(max_sr_std, max_sr_returns, color='red', s=100, edgecolors='black')

        # Min Vol
        plt.scatter(min_vol_std, min_vol_returns, color='green', s=100, edgecolors='black')

        # Efficient Frontier
        plt.plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
                 color='black', linestyle='-.', linewidth=2)

        plt.title('Efficient Frontiers')
        plt.xlabel('Annualized Volatility (%)')
        plt.ylabel('Annualized Return (%)')
        plt.legend(['Maximum Sharpe Ratio', 'Minimum Volatility', 'Historical Efficient Frontier'], loc='best')

    plt.show()
########################################################################################################################
"""
EFFICIENT FRONTIER METHOD 2 => SYSTEM RESOLUTION NEDD TO BE CHANGED IF WEIGHTS CONSTRAINTS

mean_returns_np = mean_returns.values.reshape((-1, 1))
# Convert dataframe in matrix numpy
cov_matrix_np = cov_matrix.values
# Inverse covariance matrix
cov_matrix_inv = np.linalg.inv(cov_matrix_np)

# Matrix column 1
ones_matrix = np.ones((mean_returns_np.shape[0], 1))

A = float(ones_matrix.T @ cov_matrix_inv @ mean_returns_np)
B = float(mean_returns_np.T @ cov_matrix_inv @ mean_returns_np)
C = float(ones_matrix.T @ cov_matrix_inv @ ones_matrix)
D = float(B * C - A**2)

# Weight vector : X = g + E(Rp) * h
g = (B * cov_matrix_inv @ ones_matrix - A * cov_matrix_inv @ mean_returns_np) / D
h = (C * cov_matrix_inv @ mean_returns_np - A * cov_matrix_inv @ ones_matrix) / D


def portVar(w, cov_mat):
    port_std = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(12)
    return port_std


efficient_list_std = []
target_rtrns = np.linspace(0.01, 0.35, 10)
for rtrns in target_rtrns:
    ret_0 = float((portVar(g, cov_matrix))**2)
    ret_1 = float((portVar(g + h, cov_matrix))**2)
    ret_target = float((portVar(g + rtrns * h, cov_matrix))**2)
    # System resolution
    matrix_coef = np.array([[0, 0, 1], [1, 1, 1], [rtrns**2, rtrns, 1]])
    matrix_results = np.array([ret_0, ret_1, ret_target]).reshape((-1, 1))
    abc = np.linalg.solve(matrix_coef, matrix_results)  # abc donne les val de a, b et c pour coef vriance port
    min_var_port_target_ret = float((abc[0] * rtrns**2 + abc[1] * rtrns + abc[2])*12)
    efficient_list_std.append(min_var_port_target_ret)


# Plot of EF
plt.scatter(efficient_list_std, target_rtrns)
plt.xlabel('efficient_list_var')
plt.ylabel('target_rtrns')
plt.show()
"""
########################################################################################################################
"""
EFFICIENT FRONTIER METHOD 3



def portfolio_variance(w, cov_mat):
    return np.dot(w.T, np.dot(cov_mat, w))


# fonction qui calcule la moyenne d'un portfolio
def portfolio_mean(w, m_returns):
    return np.dot(w.T, m_returns)


# contrainte pour la somme des poids égale à 1
def constraint_sum(w):
    return np.sum(w) - 1


# contrainte pour la moyenne du portfolio
def constraint_mean(w, m_returns, target_mean_port):
    return portfolio_mean(w, m_returns) - target_mean_port


# initialisation des poids (equally weighted)
weights = np.ones(mean_returns.shape[0]) / mean_returns.shape[0]

# définition des bornes pour les poids
# Vector of size number of assets, for each ones bounds is (0,1)
bounds = tuple((-1, 1) for i in range(mean_returns.shape[0]))

# définition de la contrainte de somme des poids
constraint1 = {'type': 'eq', 'fun': constraint_sum}

# Définir une plage de valeurs pour la moyenne du portfolio
mean_range = np.arange(0.01, 0.04, 0.001)

# Initialiser les listes pour stocker les résultats optimaux de la variance et de la moyenne
variances = []
means = []

# Pour chaque valeur de la moyenne du portfolio dans la plage définie, résoudre l'optimisation sous contraintes
for target_mean in mean_range:
    # Définir la contrainte de moyenne pour la valeur actuelle de la moyenne cible
    constraint2 = {'type': 'eq', 'fun': constraint_mean, 'args': (mean_returns, target_mean)}
    constraints = [constraint1, constraint2]

    # Résoudre l'optimisation sous contraintes pour trouver les poids optimaux
    results = sc.minimize(portfolio_variance, weights, args=(cov_matrix,), method='SLSQP', bounds=bounds,
                           constraints=constraints)

    # Ajouter les résultats optimaux de la variance et de la moyenne aux listes
    variances.append(portfolio_variance(results.x, cov_matrix) * 12**0.5)
    means.append(portfolio_mean(results.x, mean_returns) * 12)

    portVar = np.array(portfolio_variance(results.x, cov_matrix) * 12**0.5)
    portMean = np.array(portfolio_mean(results.x, mean_returns) * 12)

# Tracer la frontière efficiente
plt.plot(np.sqrt(variances), means)
plt.scatter(np.sqrt(portVar), portMean)
plt.ylim(0, 0.6)
plt.xlim(0, 0.12)
plt.xlabel('Annualised Standard Deviation')
plt.ylabel('Annualised Returns')
plt.title('Efficient Frontier')
plt.show()
"""
########################################################################################################################
"""
QUESTION 1.3
RESAMPLING
"""
# For iterations
nbr_new_samples = 2
nbr_target_ret = len(calculated_results(mean_returns, cov_matrix, rf=0, constraint_set=(-1, 1))[7])
# Create empty tables
total_samples_var = pd.DataFrame()
total_samples_target_ret = pd.DataFrame()
total_sample_weights_list = np.array([])
# sample_means_step3 = []
sample_means_step3 = np.empty((0, nbr_target_ret))
# sample_std_step3 = []
sample_std_step3 = np.empty((0, nbr_target_ret))
# To fix randomization
np.random.seed(2)

for nbr_sample in range(nbr_new_samples):
    # STEP 2 : Sample generation
    sample = np.random.multivariate_normal(mean_returns, cov_matrix, len(df_returns))
    sample = pd.DataFrame(sample, columns=df_returns.columns, index=df_returns.index)

    # Optimization to get optimal portfolio weights, returns and variances
    variance_new_sample, target_ret_new_sample,\
        weights_new_sample = calculated_results(sample.mean(), sample.cov(ddof=1), rf=0, constraint_set=(-1, 1))[6:9]

    # Store variances and target returns of the new samples
    variance_new_sample = pd.DataFrame(variance_new_sample)
    target_ret_new_sample = pd.DataFrame(target_ret_new_sample)
    total_samples_var = pd.concat([total_samples_var, variance_new_sample], axis=1)
    total_samples_target_ret = pd.concat([total_samples_target_ret, target_ret_new_sample], axis=1)

    # Manipulation on weights of new samples to store them
    weights_new_sample = np.array([weights_new_sample])

    if nbr_sample == 0:
        total_sample_weights_list = weights_new_sample
    else:
        total_sample_weights_list = np.vstack([total_sample_weights_list, weights_new_sample])  # Becomes 3D

    # Reduce dimension into 2D
    weights_new_sample = np.squeeze(weights_new_sample, axis=0)
    weights_new_sample = weights_new_sample.T

    # Manipulation on the original dataframe mean_returns
    mean_returns_numpy = mean_returns.to_numpy()
    mean_returns_numpy = np.reshape(mean_returns_numpy, (-1, 1))
    # mean_returns_numpy = mean_returns_numpy.T

    # STEP 3 : Efficient frontier of new samples calculated with original portfolio return and covariance
    sample_means_step3 = np.concatenate((sample_means_step3, np.dot(mean_returns_numpy.T, weights_new_sample) * 12),
                                        axis=0)
    liste = []

    for asset in range(nbr_target_ret):
        liste.append(np.sqrt(np.dot(weights_new_sample[:, asset].T,
                                    np.dot(cov_matrix, weights_new_sample[:, asset]))) * np.sqrt(12))

    liste = np.array(liste).reshape(1, -1)
    sample_std_step3 = np.append(sample_std_step3, liste, axis=0)

# STEP 4
# Add keepdims=True to keep 3D
sample_average_weights_step4 = np.mean(total_sample_weights_list[0:nbr_new_samples, :, :], axis=0).T
# Efficient frontier of average samples calculated with original portfolio return and covariance
sample_means_step4 = np.dot(mean_returns_numpy.T, sample_average_weights_step4) * 12

liste2 = []
for asset in range(nbr_target_ret):
    liste2.append(np.sqrt(np.dot(sample_average_weights_step4[:, asset].T,
                                 np.dot(cov_matrix, sample_average_weights_step4[:, asset]))) * np.sqrt(12))
liste2 = np.array(liste2).reshape(1, -1)
sample_std_step4 = np.empty((0, nbr_target_ret))
sample_std_step4 = np.append(sample_std_step4, liste2, axis=0)

# Run it just once
sample_means_step3 *= 100
sample_std_step3 *= 100
sample_means_step4 *= 100
sample_std_step4 *= 100

# 1 for Historical Efficient Frontier only and 2 for having both H.E.F and Resampled Efficient Frontier
ef_graph(2, mean_returns, cov_matrix, sample_means_step3,
         sample_std_step3, sample_means_step4, sample_std_step4, nbr_new_samples)

"LORS REDACTION 1.3, FAIRE MESURES ESTIMATION cf. SLIDE 19 LECTURE 3"
########################################################################################################################
"""
QUESTION 1.4
"""
# Weights concerning the minimum variance portfolio
weights_min_var = min_var(mean_returns, cov_matrix)["x"]


def marginal_contributions_risk(w, m_retunrs, cov_mat):
    # Marginal contributions to risk
    MCR = np.dot(cov_mat, w) / np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    # Retrieve variance of each asset from the covariance matrix
    std_devs = np.sqrt(np.diag(cov_mat))
    # Covariance of each asset with the portfolio
    cov_with_portfolio = np.zeros(len(w))
    for i in range(len(w)):
        cov_i = w[i] * std_devs[i]
        for j in range(len(w)):
            if j != i:
                cov_i += w[j] * cov_mat.iloc[i, j]
        cov_with_portfolio[i] = cov_i
    # Marginal contributions to risk => what is different with MCR ?????
    MCRi = cov_with_portfolio / min_var(m_retunrs, cov_mat)["fun"]

    return MCR, MCRi


print(marginal_contributions_risk(min_var(mean_returns, cov_matrix)["x"], mean_returns, cov_matrix))
# test MCRi et MCR
# test = cov_matrix.loc[cov_matrix.index != "STANLEY BLACK & DECKER", "MGIC INVESTMENT"]
# test = cov_matrix.iloc[1:, 0]
# test2 = weights_min_var[np.r_[0:2, 3:144]]
# test2 = weights_min_var[1:]
# print((weights_min_var[0]*std_devs[0] + np.dot(test, test2))
#       /min_var(mean_returns, cov_matrix)["fun"])


"""Global Minimum Portfolio"""
# Global Minimum Variance Portfolio attributes the same marginal volatility to all the assets


def GMP(m_returns, cov_mat):
    # Inverse covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_mat)
    # Matrix column 1
    ones_matrix = np.ones(len(m_returns))

    optimal_weights = np.dot(cov_matrix_inv, ones_matrix) / np.dot(ones_matrix.T, np.dot(cov_matrix_inv, ones_matrix))

    GMP_perf = portfolio_perf(optimal_weights, m_returns, cov_mat)

    return GMP_perf, optimal_weights


print(GMP(mean_returns, cov_matrix))
print(marginal_contributions_risk(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix))

# VAR computed on returns directly
alpha = 0.01
VAR = portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[0] - norm.ppf(1 - alpha) * \
      portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[1]


def var_parametric(portreturn, portstd, distribution="normal", alpha=1, dof=6):
    """Calculate the portfolio VaR given a distribution, with known parameters"""
    if distribution == "normal":
        VaR = portreturn - norm.ppf(1 - alpha / 100) * portstd
    elif distribution == "t-distribution":
        nu = dof
        VaR = portreturn - np.sqrt((nu - 2) / nu) * t.ppf(1 - alpha / 100, nu) * portstd
    else:
        raise TypeError("Expected distribution to be normal or t-distributed")
    return VaR


# Expected ShortFall


def cvar_parametric(portreturn, portstd, distribution="normal", alpha=1, dof=6):
    """Calculate the portfolio CVaR given a distribution, with known parameters"""
    if distribution == "normal":
        CVaR = portreturn - (alpha / 100) ** -1 * norm.pdf(norm.ppf(alpha / 100)) * portstd
    elif distribution == "t-distribution":
        nu = dof
        x_anu = t.ppf(alpha / 100, nu)
        CVaR = portreturn - -1 / (alpha / 100) * (1 - nu) ** -1 * (nu - 2 + x_anu ** 2) * t.pdf(x_anu, nu) * portstd
    else:
        raise TypeError("Expected distribution to be normal or t-distributed")
    return CVaR


normVaR = var_parametric(portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[0],
                         portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[1])
normCVaR = cvar_parametric(portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[0],
                           portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[1])
tVaR = var_parametric(portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[0],
                      portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[1], distribution="t-distribution")
tCVaR = cvar_parametric(portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[0],
                        portfolio_perf(GMP(mean_returns, cov_matrix)[1], mean_returns, cov_matrix)[1], distribution="t-distribution")

print(normVaR)
print(normCVaR)
print(tVaR)
print(tCVaR)


# Max Drowdown
df_returns_G_M_Port = df_returns * GMP(mean_returns, cov_matrix)[1]
df_returns_G_M_Port = df_returns_G_M_Port.assign(Sum=df_returns_G_M_Port.sum(axis=1))

# Tracer un graphique en courbes
plt.plot(df_returns_G_M_Port.index, df_returns_G_M_Port['Sum'])

# Définir les étiquettes des axes
plt.xlabel('Dates')
plt.ylabel('Somme des colonnes')

# Afficher le graphique
plt.show()


max_data_stock = df_returns_G_M_Port["Sum"].rolling(window=len(df_returns_G_M_Port["Sum"]), min_periods=1).max()
dd_stock = df_returns_G_M_Port["Sum"] / max_data_stock - 1
MDD_stock = dd_stock.rolling(window=len(df_returns_G_M_Port["Sum"]), min_periods=1).min()
print(MDD_stock.min()*100)
########################################################################################################################
"""
QUESTION 1.5
"""
