import streamlit as st
from fredapi import Fred
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import seaborn as sns
import numpy as np
import warnings

from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import warnings
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

st.header("Quantitative Finance project research Dashboard")
st.header("Joseph Willson - Neil Marteau - Mélanie Yakoub- Johanna Roll")

# Sidebar with parameter
st.sidebar.title("Parameters of our study for economic factors and HMM model : ")
x_eco = st.sidebar.slider("* Weight of Log(VIX) in our economic indicator", 0.0, 1.0, 0.7, step=0.1)
st.sidebar.write("Weight of Log(VIX) in our economic indicator:", round(x_eco * 100, 2), "%")
y_eco = (1 - x_eco) / 3
st.sidebar.write("Weight of 5-year, 10-year, and 30-year interest rates in our economic indicator:", round(y_eco * 100, 2), "%")
number_of_regimes = st.sidebar.slider("* Number of regimes for the HMM model", 2, 6, 4, step=1)

################################################################# Portfolios

warnings.filterwarnings('ignore')
fred = Fred(api_key='5fb3300beee792c60ddd65e93a863aa2')  # Get your API key from the FRED website

# portf1 = yf.download('^GSPC')["Adj Close"]  # S&P 500
# portf2 = yf.download('TLT')["Adj Close"]  # treasuries : ICE U.S. Treasury 20+ Year Bond Index: Focuses on long-term U.S. Treasury bonds with maturities of 20 years or more.
# portf3 = yf.download('^SPGSCI')["Adj Close"]  # commodity :  GS&P GSCI (Goldman Sachs Commodity Index): Measures the performance of a diversified basket of commodity futures contracts, including energy, agriculture, and metals.
# portf4 = yf.download('BND')["Adj Close"]  # corporate bond : Vanguard Total Bond Market Index Fund (BND)


portf1 = pd.read_csv("./data/^GSPC.csv")
portf2 = pd.read_csv("./data/TLT.csv")
portf3 = pd.read_csv("./data/^SPGSCI.csv")
portf4 = pd.read_csv("./data/BND.csv")

portf1 = portf1.set_index('Date')
portf2 = portf2.set_index('Date')   
portf3 = portf3.set_index('Date')
portf4 = portf4.set_index('Date')

portfolios = pd.concat([portf1, portf2, portf3, portf4], axis=1).dropna()
portfolios.columns = ['S&P 500', 'Treasuries index', 'Commodity index', 'Corporate Bond index']

# Calculate returns for each portfolio
returns = portfolios.pct_change().dropna()
returns.index = pd.to_datetime(returns.index)
# Streamlit app starts here
st.title('Portfolio Analysis')

# Display the distribution of returns for each portfolio with KDE
st.subheader('Distribution of Returns for Each Portfolio with KDE')

data = []

for i, col in enumerate(returns.columns):
    hist_data = np.histogram(returns[col], bins=50, density=True)
    x = (hist_data[1][:-1] + hist_data[1][1:]) / 2
    y = hist_data[0]
    trace = go.Scatter(x=x, y=y, mode='lines', name=col)
    data.append(trace)

layout = go.Layout(title='Distribution of Returns for Each Portfolio with KDE',
                   xaxis=dict(title='Returns'),
                   yaxis=dict(title='Density'))

fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)

# Display the returns of each portfolio over time
st.subheader('Returns of Each Portfolio Over Time')

data = []

for col in returns.columns:
    trace = go.Scatter(x=returns.index, y=returns[col], mode='markers', name=col, marker=dict(opacity=0.5))
    data.append(trace)

layout = go.Layout(title='Returns of Each Portfolio Over Time',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Returns'),
                   legend=dict(orientation="h"),
                   hovermode='closest')

fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)

############################################################################### Factor and economic indicator 

# Money and credit factors : Rates
DGS5 = fred.get_series('DGS5', start_date='2003-01-01', end_date='2024-01-01').pct_change().dropna()
DGS10 = fred.get_series('DGS10', start_date='2023-01-01', end_date='2024-01-01').pct_change().dropna()
DGS30 = fred.get_series('DGS30', start_date='2023-01-01', end_date='2024-01-01').pct_change().dropna()
treasury_rate = pd.concat([DGS5,DGS10,DGS30])

# Market indicators : VIX
#data = yf.download("^VIX", end = "2024-01-02")["Adj Close"]
data = pd.read_csv("./data/VIX.csv")
data = data.set_index('Date')
data.index = pd.to_datetime(data.index)

data.tail()

economic_indicator = pd.concat([DGS5, DGS10, DGS30, data], axis=1).dropna()
economic_indicator['Log_VIX'] = np.log(economic_indicator['Adj Close'])

economic_indicator.columns = ['5Y', '10Y', '30Y', 'VIX', 'Log_VIX']
st.title('Economic indicator factors analysis')

# Perform calculations
economic_indicator['Economic_Indicator'] = x_eco * economic_indicator['Log_VIX'] + y_eco * economic_indicator['5Y'] + y_eco * economic_indicator['10Y'] + y_eco * economic_indicator['30Y']
st.write(economic_indicator.head())



from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
st.subheader(f'Economic Indicator Normalized with {round(x_eco, 2)} % of Log_VIX and {round(y_eco,2)} % of 5Y, 10Y and 30Y rates')
economic_indicator['Normalized_Indicator'] = scaler.fit_transform(economic_indicator[['Economic_Indicator']])
fig = go.Figure()
fig.add_trace(go.Scatter(x=economic_indicator.index, y=economic_indicator['Log_VIX'],
                            mode='lines', name='Log_VIX'))
st.plotly_chart(fig)


############################################################################### HMM modelization

portfolios.index = pd.to_datetime(portfolios.index)

merged_data = pd.concat([data, portfolios], axis=1, join='outer').dropna()


# Define your data (e.g., VIX and yield curve data
# Create a Markov Switching Model
mod = MarkovRegression(economic_indicator["Normalized_Indicator"], k_regimes=number_of_regimes, trend='c', switching_variance=True)

# Estimate the model parameters
res = mod.fit()

# Access the estimated regimes
regime_states = res.smoothed_marginal_probabilities[0]

# Predict the most likely regime for each day
predicted_regime = res.predict()

# Create a DataFrame with dates and regime predictions
regime_df = pd.DataFrame({'Date': regime_states.index, 'Regime': predicted_regime})

# Set the date column as the index (if it's not already)
regime_df.set_index('Date', inplace=True)

# Print or use the DataFrame as needed
print(regime_df)

probability =  res.smoothed_marginal_probabilities

# Define a function to determine the state based on the conditions
def determine_state(row):
    for i in range(len(row)):
        if row[i] > 0.5:
            return i
    return None  # You can set a default value if none of the conditions are met

# Apply the function to create the 'State' column
probability['State'] = probability.apply(determine_state, axis=1)

# Display the updated DataFrame

probability['sum'] = probability.sum(axis=1)
probability['State'] = probability.iloc[:, :-2].idxmax(axis=1)

data = pd.DataFrame(data)
data.rename(columns = {'Adj Close':'VIX'}, inplace = True)
data['Regime_Label'] = probability['State'] 
data.head()

economic_indicator["Regime_Label"] = probability['State'] 
data.Regime_Label.value_counts()

import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=np.log(data['VIX']),
    mode='markers',
    marker=dict(
        size=10,
        color=data['Regime_Label'], # set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        colorbar=dict(title="Label"), # add colorbar title
        symbol='x',
        sizemode='diameter',
        showscale=True # show color scale
    ),
    name='KMeans_Label'
))

fig.update_layout(
    title='State segmentation of our economic indicator based on HMM',
    xaxis_title='Date',
    yaxis_title='Log-(Economic Indicator)',
    legend_title='KMeans_Label',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig)


#data["Regime_Label"] = data["Regime_Label"].replace({0: 'Regime 1', 1: 'Regime 2', 2: 'Regime 3', 3: 'Regime 4'})

# Get the unique values in "Regime_Label"
unique_regimes = data["Regime_Label"].unique()

# Create a replacement dictionary
replacement_dict = {regime: f'Regime {i+1}' for i, regime in enumerate(sorted(unique_regimes))}

# Replace the values in "Regime_Label"
data["Regime_Label"] = data["Regime_Label"].replace(replacement_dict)

# Define a color map for the labels
#label_colors = {
#    'Regime 1': '#4169E1',
#    'Regime 2': 'orange',
#    'Regime 3': 'green',
#    'Regime 4': 'red',
#}
# Define a list of colors
colors = ['#4169E1', 'orange', 'green', 'red', 'purple', 'yellow', 'cyan', 'magenta']

# Assuming data is your DataFrame and 'Regime_Label' is the column with different regimes

# Get the unique values in "Regime_Label"
unique_regimes = data["Regime_Label"].unique()

# Create a color map
label_colors = {f'Regime {str(i+1)}': color for i, color in zip(range(len(unique_regimes)), colors)}

# Create a Streamlit app
# Create traces for each unique regime label
traces = []
for label in data['Regime_Label'].unique():
    x = np.log(data[data['Regime_Label'] == label]['VIX'])
    trace = go.Histogram(x=x, histnorm='probability density', opacity=0.6, autobinx=False,
                         xbins=dict(start=np.min(x), end=np.max(x), size=(np.max(x) - np.min(x)) / 50),
                         marker_color=label_colors.get(label, 'pastel'), name=label)
    traces.append(trace)

layout = go.Layout(title="Our economic indicator regime distribution based on HMM",
                   xaxis=dict(title="Economic Indicator"),
                   yaxis=dict(title="Density"),
                   barmode='overlay',
                   bargap=0.1)

fig = go.Figure(data=traces, layout=layout)
st.plotly_chart(fig)


# Assuming data is a DataFrame you have already loaded
data = pd.DataFrame(data)
data.rename(columns = {'Adj Close':'VIX'}, inplace = True)
data['Regime_Label'] = probability['State'] 

merged_data = merged_data.dropna()
data_returns = merged_data[['S&P 500', 'Treasuries index', 'Commodity index', 'Corporate Bond index']].pct_change().dropna()
data_returns["Regime_Label"] = data["Regime_Label"]


from scipy import stats

# Create Streamlit app
st.title("Kolmogorov-Smirnov Test for Different Regimes")

data_returns = data_returns.dropna()
replacement_dict = {regime: f'Regime {i+1}' for i, regime in enumerate(sorted(data_returns['Regime_Label'].unique()))}
data_returns["Regime_Label_remap"] = data_returns["Regime_Label"].replace(replacement_dict)

st.write(data_returns)

# Replace the values in "Regime_Label"

# Allow user to select columns and regimes
selected_column = st.selectbox("Select a column:", data_returns.iloc[:, :4].columns)
selected_regime_1 = st.selectbox("Select first regime:", unique_regimes)
selected_regime_2 = st.selectbox("Select second regime:", unique_regimes)

# Perform Kolmogorov-Smirnov test
statistic, p_value = stats.ks_2samp(data_returns[data_returns["Regime_Label_remap"] == selected_regime_1][selected_column],
                                    data_returns[data_returns["Regime_Label_remap"] == selected_regime_2][selected_column])

alpha = 0.05
if p_value < alpha:
    result = "The two distributions are significantly different."
else:
    result = "The two distributions are not significantly different."

# Display the result
st.write(f"Kolmogorov-Smirnov Test Statistic: {statistic}")
st.write(f"P-value: {p_value}")
st.write(result)

del data_returns["Regime_Label"]
import plotly.graph_objects as go
import streamlit as st

import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

# Assuming your DataFrame is named data
for column in data_returns.columns:
    if column != 'Regime_Label_remap':
        fig = go.Figure()
        for regime in data_returns['Regime_Label_remap'].unique():
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data_returns[data_returns['Regime_Label_remap'] == regime][column],
                name=str(regime),
                nbinsx=50,
                histnorm='probability density'
            ))

            # Calculate KDE
            data = data_returns[data_returns['Regime_Label_remap'] == regime][column]
            kde = gaussian_kde(data)
            kde_x = np.linspace(data.min(), data.max(), 1000)
            kde_y = kde.evaluate(kde_x)

            # Add KDE line plot
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE for ' + str(regime)
            ))

        fig.update_layout(
            title_text="Distribution of Returns for " + column,
            xaxis_title_text=column,
            yaxis_title_text='Density',
            bargap=0.2,
            bargroupgap=0.1
        )
        st.plotly_chart(fig)


############################################################################### Factor analysis by HMM regime
# Assuming your DataFrame is named data

del economic_indicator["Regime_Label"]
df_estimations = pd.concat([data_returns, economic_indicator], axis=1, join='outer').dropna()

# Step 3: Split Data by Regime
regime_groups = df_estimations.groupby('Regime_Label_remap')


factors = ['5Y', '10Y', '30Y', 'Log_VIX']

st.title("Factor Analysis of our portfolios in one single state regime : ")

for target in ['S&P 500', 'Treasuries index', 'Commodity index', 'Corporate Bond index']:

    st.subheader(f"Factor Analysis for {target} by Regime")

    # Step 4: Factor Estimation
    X = df_estimations[factors]  # Independent variables (factors)
    X = sm.add_constant(X)  # Add constant term for intercept
    y = df_estimations[target]  # Dependent variable (S&P 500 returns)
    
    # Fit OLS regression model
    model = sm.OLS(y, X)
    results = model.fit()
    # Extract log likelihood, R-squared, and adjusted R-squared
    log_likelihood = results.llf
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    significant_coefficients = results.pvalues[results.pvalues < 0.1].index.tolist()
    
    sentence = f"In the one single state regime, the log likelihood is {log_likelihood:.2f}, R-squared is {r_squared:.2f}, adjusted R-squared is {adj_r_squared:.2f}, and the following coefficients are statistically significant at the 10% level: {', '.join(significant_coefficients)}."
    
    st.write(sentence)
    st.table(results.summary2().tables[1])  # Display only the coefficients table





st.title("Factor Analysis of our portfolios by Regime (HMM) :")

for target in ['S&P 500', 'Treasuries index', 'Commodity index', 'Corporate Bond index']:

    st.subheader(f"Factor Analysis for {target} by Regime")

    # Step 4: Factor Estimation
    for regime, group in regime_groups:
        X = group[factors]  # Independent variables (factors)
        X = sm.add_constant(X)  # Add constant term for intercept
        y = group[target]  # Dependent variable (S&P 500 returns)
        
        # Fit OLS regression model
        model = sm.OLS(y, X)
        results = model.fit()
        # Extract log likelihood, R-squared, and adjusted R-squared
        log_likelihood = results.llf
        r_squared = results.rsquared
        adj_r_squared = results.rsquared_adj
        significant_coefficients = results.pvalues[results.pvalues < 0.1].index.tolist()
        
        sentence = f"In the {regime} regime, the log likelihood is {log_likelihood:.2f}, R-squared is {r_squared:.2f}, adjusted R-squared is {adj_r_squared:.2f}, and the following coefficients are statistically significant at the 10% level: {', '.join(significant_coefficients)}."
        
        st.write(sentence)
        st.write(f"Regime: {regime}")
        st.table(results.summary2().tables[1])  # Display only the coefficients table


del df_estimations["Regime_Label_remap"]
del df_estimations["Normalized_Indicator"]

import plotly.figure_factory as ff

# Assuming your DataFrame is named df_estimations
correlation_matrix = df_estimations.corr()

st.subheader("Annexe : Correlation Matrix of our economic indicator and portfolios")

fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    annotation_text=correlation_matrix.round(2).values,
    showscale=True,
    colorscale='Viridis'
)

st.plotly_chart(fig)




