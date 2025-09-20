import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import seaborn as sns

st.set_page_config(layout='wide', page_title='Startup Funding Dashboard')

# Load and preprocess data
df = pd.read_csv('startup_cleaned.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['city'] = df['city'].fillna('Unknown')
df['vertical'] = df['vertical'].fillna('Other')

# Split investors for word cloud & filters
df['investor_list'] = df['investors'].fillna('').str.split(',')

# Flatten list of investors for wordcloud and filters
investor_flat = [inv.strip() for sublist in df['investor_list'] for inv in sublist if inv.strip() != '']

# Sidebar filters for startup
st.sidebar.header("Filter Startups")
selected_vertical = st.sidebar.multiselect('Vertical(s)', options=sorted(df['vertical'].unique()), default=None)
selected_city = st.sidebar.multiselect('City(s)', options=sorted(df['city'].unique()), default=None)

def filter_startups(data):
    if selected_vertical:
        data = data[data['vertical'].isin(selected_vertical)]
    if selected_city:
        data = data[data['city'].isin(selected_city)]
    return data

def load_overall_analysis():
    st.title('Overall Funding Analysis')

    filtered_df = filter_startups(df)

    # Basic Metrics
    total = round(filtered_df['amount'].sum())
    max_funding = filtered_df.groupby('startup')['amount'].max().max()
    avg_funding = filtered_df.groupby('startup')['amount'].sum().mean()
    num_startups = filtered_df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Investment', f'{total} Cr')
    col2.metric('Max Investment', f'{round(max_funding)} Cr')
    col3.metric('Avg Investment', f'{round(avg_funding)} Cr')
    col4.metric('Funded Startups', num_startups)

    # Top 5 Startups by Funding
    st.subheader("Top 5 Funded Startups")
    top_startups = filtered_df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots()
    sns.barplot(x=top_startups.values, y=top_startups.index, palette='viridis', ax=ax)
    ax.set_xlabel('Total Funding (Cr)')
    ax.set_ylabel('Startup')
    st.pyplot(fig)

    # Funding Round Distribution
    st.subheader("Funding Round Distribution")
    round_counts = filtered_df['round'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(round_counts.values, labels=round_counts.index, autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    st.pyplot(fig2)

    # MoM Funding Trend with Rolling Average
    st.subheader("Monthly Funding Trend")
    monthly_df = filtered_df.groupby(['year', 'month'])['amount'].sum().reset_index()
    monthly_df['date'] = pd.to_datetime(monthly_df[['year', 'month']].assign(DAY=1))
    monthly_df = monthly_df.sort_values('date')
    monthly_df['rolling_avg'] = monthly_df['amount'].rolling(window=3, min_periods=1).mean()

    fig3, ax3 = plt.subplots()
    ax3.plot(monthly_df['date'], monthly_df['amount'], label='Monthly Total')
    ax3.plot(monthly_df['date'], monthly_df['rolling_avg'], label='3-Month Rolling Avg', linestyle='--')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Funding (Cr)')
    ax3.legend()
    st.pyplot(fig3)

def load_startup_analysis():
    selected_startup = st.sidebar.selectbox('Select Startup', sorted(df['startup'].unique()))
    st.title(f"Startup Analysis: {selected_startup}")
    startup_df = df[df['startup'] == selected_startup]

    st.write(f"Total Funding: {startup_df['amount'].sum()} Cr")
    st.write(f"Funding Rounds: {startup_df.shape[0]}")
    st.dataframe(startup_df[['date', 'round', 'amount', 'investors', 'vertical', 'city']])

    # Funding over time for selected startup
    funding_over_time = startup_df.groupby(['year', 'month'])['amount'].sum().reset_index()
    funding_over_time['date'] = pd.to_datetime(funding_over_time[['year', 'month']].assign(DAY=1))
    fig, ax = plt.subplots()
    ax.plot(funding_over_time['date'], funding_over_time['amount'], marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Funding (Cr)')
    ax.set_title('Funding Over Time')
    st.pyplot(fig)

def load_investor_details(investor):
    st.title(f"Investor Analysis: {investor}")

    inv_df = df[df['investors'].str.contains(investor, na=False)]

    st.subheader('Recent Investments')
    recent_df = inv_df.sort_values(by='date', ascending=False).head()
    st.dataframe(recent_df[['date', 'startup', 'vertical', 'city', 'round', 'amount']])

    col1, col2 = st.columns(2)

    with col1:
        biggest_investments = inv_df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(5)
        st.subheader('Biggest Investments')
        fig, ax = plt.subplots()
        sns.barplot(x=biggest_investments.values, y=biggest_investments.index, palette='magma', ax=ax)
        ax.set_xlabel('Investment Amount (Cr)')
        ax.set_ylabel('Startup')
        st.pyplot(fig)

    with col2:
        vertical_investments = inv_df.groupby('vertical')['amount'].sum()
        st.subheader('Sector Distribution')
        fig2, ax2 = plt.subplots()
        ax2.pie(vertical_investments.values, labels=vertical_investments.index, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

    # YoY Investment Trend
    yearly_inv = inv_df.groupby('year')['amount'].sum()
    st.subheader('Yearly Investment Trend')
    fig3, ax3 = plt.subplots()
    ax3.plot(yearly_inv.index, yearly_inv.values, marker='o')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Investment Amount (Cr)')
    st.pyplot(fig3)

def load_leaderboards():
    st.title("Leaderboards")

    # Top Investors by Total Investment
    inv_totals = pd.Series(investor_flat).value_counts()
    inv_sums = df.explode('investor_list').groupby('investor_list')['amount'].sum()
    top_investors = inv_sums.sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Investors by Funding")
        fig, ax = plt.subplots()
        sns.barplot(x=top_investors.values, y=top_investors.index, palette='cool', ax=ax)
        ax.set_xlabel('Total Investment (Cr)')
        st.pyplot(fig)

    with col2:
        st.subheader("Investor Activity Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(investor_flat))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

def load_map_view():
    st.title("Funding Map by City")

    city_data = df.groupby('city')['amount'].sum().reset_index()
    city_data = city_data[city_data['city'] != 'Unknown']

    try:
        import pydeck as pdk

        # You need a city-lat-long dataset or geocoder here; assuming we have lat/lon columns
        # For demo, we create random lat/lon near India (adjust as per your data)
        np.random.seed(42)
        city_data['lat'] = np.random.uniform(20, 30, size=city_data.shape[0])
        city_data['lon'] = np.random.uniform(70, 80, size=city_data.shape[0])

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=24.5,
                longitude=77.5,
                zoom=5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=city_data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius='amount * 10000',
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
            tooltip={"text": "{city}\nFunding: {amount} Cr"}
        ))
    except ImportError:
        st.error("pydeck is required for map visualization. Install it with `pip install pydeck`.")

# Main menu in sidebar
option = st.sidebar.selectbox("Choose Analysis", ['Overall Analysis', 'Startup Analysis', 'Investor Analysis', 'Leaderboards', 'Map View'])

if option == 'Overall Analysis':
    load_overall_analysis()
elif option == 'Startup Analysis':
    load_startup_analysis()
elif option == 'Investor Analysis':
    # Select investor with dropdown and button to load details
    unique_investors = sorted(set(investor_flat))
    selected_investor = st.sidebar.selectbox('Select Investor', unique_investors)
    btn = st.sidebar.button('Load Investor Details')
    if btn:
        load_investor_details(selected_investor)
elif option == 'Leaderboards':
    load_leaderboards()
else:
    load_map_view()
