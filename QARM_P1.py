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
# df_returns = df_returns.fillna(df_returns.mean())
cov_matrix = df_returns.cov()
# cov_matrix = df_returns.cov(min_periods=72)
corr_matrix = df_returns.corr()

# TEST COV_MATRIX
# determinant = print(np.linalg.det(cov_matrix))
# eig_values, eig_vectors = np.linalg.eig(cov_matrix)
# if np.all(eig_values > 0):
#     print("La matrice est définie positive.")
# else:
#     print("La matrice n'est pas définie positive.")


# tracer le graphique pour chaque actif
for column in df_returns.columns:
    plt.plot(df_returns.index, df_returns[column], label=column)

# ajouter les titres et légendes
plt.title('Rendements des actifs')
plt.xlabel('Dates')
plt.ylabel('Rendements')
# plt.legend(loc='upper left')

# afficher le graphique
plt.show()
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
    target_return = np.linspace(min_vol_returns, max_sr_returns, 10)
    for target in target_return:
        efficient_list.append(efficient_optimization(m_returns, cov_mat, target)["fun"])

    max_sr_returns, max_sr_std = round(max_sr_returns * 100, 2), round(max_sr_std * 100, 2)
    min_vol_returns, min_vol_std = round(min_vol_returns * 100, 2), round(min_vol_std * 100, 2)

    return max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return


# print(calculated_results(mean_returns, cov_matrix))
# print(calculated_results(mean_returns, cov_matrix))

"""
def ef_graph(m_returns, cov_mat, rf=0, constraint_set=(-1, 1)):
    # Return a graph ploting the min vol, max SR and efficient frontier
    max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return = calculated_results(m_returns, cov_mat, rf, constraint_set)

    # Max SR
    max_sr_rat = go.Scatter(
        name="Maximum Sharpe Ratio",
        mode="markers",
        x=[max_sr_std],
        y=[max_sr_returns],
        marker=dict(color="red", size=14, line=dict(width=3, color="black")))

    # Min Vol
    min_vol = go.Scatter(
        name="Minimum Volatility",
        mode="markers",
        x=[min_vol_std],
        y=[min_vol_returns],
        marker=dict(color="green", size=14, line=dict(width=3, color="black")))

    # Efficient Frontier
    ef_curve = go.Scatter(
        name="Efficient Frontier",
        mode="markers",
        x=[round(ef_std * 100, 2) for ef_std in efficient_list],
        y=[round(target * 100, 2) for target in target_return],
        line=dict(color="black", width=4, dash="dashdot"))

    data = [max_sr_rat, min_vol, ef_curve]

    layout = go.Layout(
        title="Portfolio Optimisation with the Efficient Frontier",
        yaxis=dict(title="Annualized Return (%)"),
        xaxis=dict(title="Annualized Volatility (%)"),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder="normal",
            bgcolor="#E2E2E2",
            bordercolor="black",
            borderwidth=2),
        width=800,
        height=600)

    fig = go.Figure(data=data, layout=layout)

    return fig.show()


ef_graph(mean_returns, cov_matrix)
"""


def ef_graph(m_returns, cov_mat, rf=0, constraint_set=(-1, 1)):
    """Return a graph ploting the min vol, max SR and efficient frontier"""
    max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return = calculated_results(m_returns, cov_mat, rf, constraint_set)

    # Max SR
    plt.scatter(max_sr_std, max_sr_returns, color='red', s=100, edgecolors='black')

    # Min Vol
    plt.scatter(min_vol_std, min_vol_returns, color='green', s=100, edgecolors='black')

    # Efficient Frontier
    plt.plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
             color='black', linestyle='-.', linewidth=2)

    plt.title('Portfolio Optimization with the Efficient Frontier')
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.legend(['Maximum Sharpe Ratio', 'Minimum Volatility', 'Efficient Frontier'], loc='best')

    plt.show()


ef_graph(mean_returns, cov_matrix)


# Autre moyen
# w_alpha_min = (1 - (np.dot(np.ones(len(mean_returns)), np.dot(np.linalg.inv(cov_matrix), mean_returns))/1))
# alpha_min = np.dot(np.linalg.inv(cov_matrix), np.ones(len(mean_returns))) /\
#             np.dot(np.ones(len(mean_returns)), np.dot(np.linalg.inv(cov_matrix), np.ones(len(mean_returns))))
# w_alpha_spec = (np.dot(np.ones(len(mean_returns)), np.dot(np.linalg.inv(cov_matrix), mean_returns))/1)
# alpha_spec = np.dot(np.linalg.inv(cov_matrix), mean_returns) /\
#             np.dot(np.ones(len(mean_returns)), np.dot(np.linalg.inv(cov_matrix), mean_returns))
# weight_formule_cours = w_alpha_min * alpha_min + w_alpha_spec * alpha_spec
#
# portfolio_perf(weight_formule_cours, mean_returns, cov_matrix)
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

# Create empty tables
num_samples = 10
sample_means = []
sample_covs = []
sample_weights = []

np.random.seed(10)
for i in range(num_samples):
    # Generate x samples
    sample = np.random.multivariate_normal(mean_returns, cov_matrix, len(df_returns))
    sample = pd.DataFrame(sample, columns=df_returns.columns, index=df_returns.index)

    # Get mean and cov of samples
    sample_mean_returns = sample.mean()
    sample_cov = sample.cov(ddof=1)

    # Add it to a list
    sample_means.append(sample_mean_returns)
    sample_covs.append(sample_cov)

##########################
"PLOT ALL GENERATED SAMPLES EFFICIENT FRONTIERS"


def ef_sample_graph(samples_means, samples_covs, rf=0, constraint_set=(-1, 1), colors=None, linestyles=None):
    plt.figure(figsize=(7, 5))
    for i, (m_returns, cov_mat) in enumerate(zip(samples_means, samples_covs)):
        max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
            efficient_list, target_return = calculated_results(m_returns, cov_mat, rf, constraint_set)

        color = colors[i] if colors else None
        linestyle = linestyles[i] if linestyles else None

        # Max SR
        plt.scatter(max_sr_std, max_sr_returns, color='red', s=100, edgecolors='black')

        # Min Vol
        plt.scatter(min_vol_std, min_vol_returns, color='green', s=100, edgecolors='black')

        # Efficient Frontier
        plt.plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
                 color=color, linestyle=linestyle, linewidth=2)

    plt.ylim(0, 60)
    plt.xlim(0, 10)
    plt.title('Sampled Portfolios Efficient Frontiers')
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    # plt.legend([' Maximum Sharpe Ratio', 'Minimum Volatility', 'Efficient Frontier'], loc='best')

    plt.show()


ef_sample_graph(sample_means, sample_covs)  # , colors=['blue', 'red'], linestyles=['-', '--']


# Hint: How to find weights for a targeted return portfolio of 20%: efficient_optimization(mean_returns,
# cov_matrix, 0.2)["x"]

# Stocks optimal weights of historical returns for m portfolios
def stock_weights(mean, cov):
    # Get weights of x portfolio equally spaced
    # Find the minimum variance portfolio weights
    min_var_portfolio = min_var(mean, cov)
    min_var_weights = min_var_portfolio['x']

    # Find the maximum Sharpe ratio portfolio weights
    max_sharpe_portfolio = max_sharpe_ratio(mean, cov)
    max_sharpe_weights = max_sharpe_portfolio['x']

    # Find the weights of the other x-2 equally spaced portfolios
    m_portfolios = 10  # checked and no differences between 10 and 30 portfolios
    target_returns = np.linspace(port_return(min_var_weights, mean, cov),
                                 port_return(max_sharpe_weights, mean, cov),
                                 m_portfolios)[1:-1]
    portfolio_weights = [min_var_weights, max_sharpe_weights]

    for target_return in target_returns:
        portfolio = efficient_optimization(mean, cov, target_return)['x']
        portfolio_weights.append(portfolio)
    return portfolio_weights


portfolio_weights = stock_weights(mean_returns,
                                  cov_matrix)
# Get a list where table 0 is weights in companies for MVP. Index 9 is weights in companies for Max SR.


# Stocks optimal weights of generated sample returns for m portfolios
def stock_Sample_weights(sample_means, sample_covs):
    portfolio_weights = []
    num_samples = len(sample_means)
    m_portfolios = 10  # checked and no differences between 10 and 30 portfolios

    for i in range(num_samples):
        mean = sample_means[i]
        cov = sample_covs[i]

        # Find the minimum variance portfolio weights
        min_var_portfolio = min_var(mean, cov)
        min_var_weights = min_var_portfolio['x']

        # Find the maximum Sharpe ratio portfolio weights
        max_sharpe_portfolio = max_sharpe_ratio(mean, cov)
        max_sharpe_weights = max_sharpe_portfolio['x']

        # Find the weights of the other x-2 equally spaced portfolios
        target_returns = np.linspace(port_return(min_var_weights, mean, cov),
                                     port_return(max_sharpe_weights, mean, cov),
                                     m_portfolios)[1:-1]
        sample_portfolio_weights = [min_var_weights, max_sharpe_weights]

        for target_return in target_returns:
            portfolio = efficient_optimization(mean, cov, target_return)['x']
            sample_portfolio_weights.append(portfolio)

        portfolio_weights.append(sample_portfolio_weights)

    return portfolio_weights


portfolio_weights = stock_Sample_weights(sample_means, sample_covs)

# Create an array containing average between sample weights for each entreprise and portfolio.
num_portfolios = len(portfolio_weights[0])

# Initialization of an array to store average weights
mean_weights = np.zeros((num_portfolios, len(portfolio_weights[0][0])))

for i in range(num_portfolios):
    # Loop over companies
    for j in range(len(portfolio_weights[0][i])):
        # Calculation of average weights for company j and portfolio i
        weights_j_i = [portfolio_weights[k][i][j] for k in range(len(portfolio_weights))]
        mean_weights[i][j] = np.mean(weights_j_i)

# Create an array containing returns and volatility of averaged samples for different portfolios.
resampled_port_perf = []
for i in range(0, len(mean_weights)):
    resampled_port_perf.append(portfolio_perf(mean_weights[i], mean_returns, cov_matrix))

resampled_port_perf = np.array(resampled_port_perf)

# Sort from lowest return to highest returns
# Sorting index in relation to column 0
sort_index = np.argsort(resampled_port_perf[:, 0])
# Table sorted by column 0
resampled_port_perf = resampled_port_perf[sort_index]

"PLOT SINGLE EFFICIENT FRONTIER OF A WEIGHTED PORTFOLIO OF ALL SAMPLED PORTFOLIOS"
plt.figure(figsize=(7, 5))
# Max SR
plt.scatter(np.max(resampled_port_perf[:, 1]) * 100, np.max(resampled_port_perf[:, 0]) * 100, color='red', s=100,
            edgecolors='black')  # *100 to get in %
# Min Vol
plt.scatter(np.min(resampled_port_perf[:, 1]) * 100, np.min(resampled_port_perf[:, 0]) * 100, color='green', s=100,
            edgecolors='black')  # *100 to get in %

plt.plot(resampled_port_perf[:, 1] * 100, resampled_port_perf[:, 0] * 100, color='blue', linestyle='-',
         linewidth=2)  # *100 to get in %
plt.ylim(0, 60)
plt.xlim(0, 10)
plt.title('Resampled Portfolio Efficient Frontier ')
plt.xlabel('Annualized Volatility (%)')
plt.ylabel('Annualized Return (%)')
plt.legend(['Maximum Sharpe Ratio Portfolio', 'Minimum Variance Portfolio', 'Efficient Frontier'], loc='best')

plt.show()

############################################################################################
"PLOT ALL EFFICIENT FRONTIERS TOGETHER"


def combined_graph(mean_returns, cov_matrix, sample_means, sample_covs, rf=0, constraint_set=(-1, 1)):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))  # 16,6

    # First subplot: Efficient Frontier for mean_returns and cov_matrix
    max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
        efficient_list, target_return = calculated_results(mean_returns, cov_matrix, rf, constraint_set)
    # Max SR
    axs[0].scatter(max_sr_std, max_sr_returns, color='red', s=100, edgecolors='black')
    # Min Vol
    axs[0].scatter(min_vol_std, min_vol_returns, color='green', s=100, edgecolors='black')
    # Efficient Frontier
    axs[0].plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
                color='black', linestyle='-', linewidth=2)

    # First subplot: Averaged Resampled Efficient Frontier
    # Max SR
    plt.scatter(np.max(resampled_port_perf[:, 1]) * 100, np.max(resampled_port_perf[:, 0]) * 100, color='red', s=100,
                edgecolors='black')  # *100 to get in %
    # Min Vol
    plt.scatter(np.min(resampled_port_perf[:, 1]) * 100, np.min(resampled_port_perf[:, 0]) * 100, color='green', s=100,
                edgecolors='black')  # *100 to get in %
    axs[0].plot(resampled_port_perf[:, 1] * 100, resampled_port_perf[:, 0] * 100, color='blue', linestyle='-',
                linewidth=2)  # *100 to get in %

    # Set parameters for plots
    axs[0].set_ylim(0, 60)
    axs[0].set_xlim(0, 10)
    axs[0].set_title('Historical & Resampled Portfolios Efficient Frontiers ')
    axs[0].set_xlabel('Annualized Volatility (%)')
    axs[0].set_ylabel('Annualized Return (%)')
    axs[0].legend(['Maximum Sharpe Ratio Portfolio', 'Minimum Variance Portfolio', 'Efficient Frontier'], loc='best')

    # Second subplot: Efficient Frontier for sample_means and sample_covs
    for i, (m_returns, cov_mat) in enumerate(zip(sample_means, sample_covs)):
        max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, \
            efficient_list, target_return = calculated_results(m_returns, cov_mat, rf, constraint_set)
        color = plt.cm.get_cmap('tab20')(i)
        # Max SR
        # axs[1].scatter(max_sr_std, max_sr_returns, color=color, s=100, edgecolors='black')
        # Min Vol
        # axs[1].scatter(min_vol_std, min_vol_returns, color=color, s=100, edgecolors='black')
        # Efficient Frontier
        axs[1].plot([ef_std * 100 for ef_std in efficient_list], [target * 100 for target in target_return],
                    color=color, linestyle='-', linewidth=2)

    # Second subplot: Averaged Resampled Efficient Frontier
    # Max SR
    plt.scatter(np.max(resampled_port_perf[:, 1]) * 100, np.max(resampled_port_perf[:, 0]) * 100, color='red', s=100,
                edgecolors='black')  # *100 to get in %
    # Min Vol
    plt.scatter(np.min(resampled_port_perf[:, 1]) * 100, np.min(resampled_port_perf[:, 0]) * 100, color='green', s=100,
                edgecolors='black')  # *100 to get in %
    axs[1].plot(resampled_port_perf[:, 1] * 100, resampled_port_perf[:, 0] * 100, color='blue', linestyle='-',
                linewidth=2)  # *100 to get in %

    # Set parameters for plots
    axs[1].set_ylim(0, 60)
    axs[1].set_xlim(0, 10)
    axs[1].set_title('Samples & Resampled Portfolios Efficient Frontiers')
    axs[1].set_xlabel('Annualized Volatility (%)')
    axs[1].set_ylabel('Annualized Return (%)')
    axs[1].legend(['Maximum Sharpe Ratio Portfolio', 'Minimum Variance Portfolio', 'Efficient Frontier'], loc='best')

    plt.show()


combined_graph(mean_returns, cov_matrix, sample_means, sample_covs)
########################################################################################################################

"""
QUESTION 1.4
"""
# Weights concerning the minimum variance portfolio
weights_min_var = min_var(mean_returns, cov_matrix)["x"]


def marginal_contributions_risk(w, m_retunrs, cov_mat):
    # Marginal contributions to risk
    MCR = np.dot(cov_mat, w) / np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    # Retrieve variance of each asset from the covaraince matrix
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

