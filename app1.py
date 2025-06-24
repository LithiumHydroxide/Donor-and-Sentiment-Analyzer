import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AMREF Donor & Sentiment Analyzer",
    page_icon="🌍",
    layout="wide"
)

# Load data (in a real app, this would be from the Excel files)
def load_data():
    # Public & Institutional Donors
    dac_countries = [
        "United States", "Japan", "United Kingdom", "France", "Germany", 
        "Denmark", "Sweden", "Canada", "Korea", "Italy", "Norway", "Belgium", 
        "Switzerland", "Ireland", "Finland", "Netherlands", "Australia", 
        "Poland", "Hungary", "Austria", "Slovak Republic", "Greece", 
        "New Zealand", "Estonia", "Luxembourg", "Iceland", "Slovenia", 
        "Czechia", "Portugal", "Lithuania", "Spain"
    ]
    
    non_dac_countries = [
        "Qatar", "United Arab Emirates", "Türkiye", "Israel", "Latvia", 
        "Thailand", "Malta", "Romania", "Kuwait", "Chinese Taipei", 
        "Croatia", "Cyprus"
    ]
    
    multilaterals = [
        "International Monetary Fund", "EU Institutions", 
        "African Development Bank", "United Nations", "IFAD", "UNICEF", 
        "UNHCR", "Central Emergency Response Fund [CERF]", "UNFPA", "UNDP", 
        "World Health Organisation [WHO]", "WFP", 
        "UN Development Coordination Office", 
        "International Labour Organisation [ILO]", 
        "Food and Agriculture Organisation [FAO]", 
        "International Atomic Energy Agency [IAEA]", "UN Women", 
        "Joint Sustainable Development Goals Fund [Joint SDG Fund]", "UNAIDS", 
        "UN Peacebuilding Fund [UNPBF]", 
        "WHO-Strategic Preparedness and Response Plan [SPRP]", 
        "WTO - International Trade Centre [ITC]", 
        "United Nations Industrial Development Organization [UNIDO]", "UNEP",
        "World Bank", "Global Fund", 
        "Global Alliance for Vaccines and Immunization [GAVI]", 
        "Global Environment Facility [GEF]", "Green Climate Fund [GCF]", 
        "Climate Investment Funds [CIF]", "Adaptation Fund", 
        "OPEC Fund for International Development [OPEC Fund]", 
        "Global Green Growth Institute [GGGI]", 
        "International Centre for Genetic Engineering and Biotechnology [ICGEB]", 
        "Nordic Development Fund [NDF]"
    ]
    
    private_donors = [
        "Bill & Melinda Gates Foundation", "Mastercard Foundation", 
        "Wellcome Trust", "IKEA Foundation", 
        "Children's Investment Fund Foundation", 
        "Conrad N. Hilton Foundation", "Ford Foundation", 
        "William and Flora Hewlett Foundation", 
        "Gatsby Charitable Foundation", "Susan T. Buffett Foundation", 
        "Rockefeller Foundation", "Charity Projects Ltd (Comic Relief)", 
        "UBS Optimus Foundation", "World Diabetes Foundation", 
        "Postcode Lottery Group", "LEGO Foundation", "Bezos Earth Fund", 
        "Open Society Foundations", 
        "John D. and Catherine T. MacArthur Foundation", 
        "Dutch Postcode Lottery", "Fondation Botnar", "Good Ventures Foundation", 
        "Margaret A. Cargill Foundation", "Bloomberg Family Foundation", 
        "Swedish Postcode Lottery", "David and Lucile Packard Foundation", 
        "Omidyar Network Fund, Inc.", "Arcus Foundation", "McKnight Foundation", 
        "Leona M. and Harry B. Helmsley Charitable Trust", "Citi Foundation", 
        "Oak Foundation", "People's Postcode Lottery", 
        "La Caixa Banking Foundation", "Jacobs Foundation", 
        "Bernard van Leer Foundation", "H&M Foundation", "MAVA Foundation", 
        "German Postcode Lottery"
    ]
    
    # Generate synthetic donation data
    np.random.seed(42)
    all_donors = dac_countries + non_dac_countries + multilaterals + private_donors
    sectors = ["Primary Health Care", "Climate", "Education", "Livelihoods"]
    regions = ["North America", "Europe", "Middle East", "Further East"]
    
    data = []
    for _ in range(1000):
        donor = np.random.choice(all_donors)
        amount = np.random.randint(10000, 1000000)
        year = np.random.choice([2021, 2022, 2023])
        sector = np.random.choice(sectors)
        
        # Assign region based on donor type
        if donor in dac_countries:
            if donor in ["United States", "Canada"]:
                region = "North America"
            elif donor in ["Japan", "Korea", "Australia"]:
                region = "Further East"
            else:
                region = "Europe"
        elif donor in non_dac_countries:
            if donor in ["Qatar", "United Arab Emirates", "Israel", "Kuwait"]:
                region = "Middle East"
            else:
                region = "Europe"
        elif donor in private_donors:
            region = np.random.choice(regions)
        else:  # multilaterals
            region = "Global"
            
        data.append({
            "Donor": donor,
            "Amount": amount,
            "Year": year,
            "Sector": sector,
            "Region": region,
            "Donor Type": "Public" if donor in (dac_countries + non_dac_countries) 
                          else "Multilateral" if donor in multilaterals 
                          else "Private"
        })
    
    return pd.DataFrame(data)

# Load data
df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
selected_years = st.sidebar.multiselect(
    "Select Years", 
    options=sorted(df['Year'].unique()), 
    default=sorted(df['Year'].unique())
)

selected_sectors = st.sidebar.multiselect(
    "Select Sectors", 
    options=df['Sector'].unique(), 
    default=df['Sector'].unique()
)

selected_regions = st.sidebar.multiselect(
    "Select Regions", 
    options=df['Region'].unique(), 
    default=df['Region'].unique()
)

selected_donor_types = st.sidebar.multiselect(
    "Select Donor Types", 
    options=df['Donor Type'].unique(), 
    default=df['Donor Type'].unique()
)

# Apply filters
filtered_df = df[
    (df['Year'].isin(selected_years)) &
    (df['Sector'].isin(selected_sectors)) &
    (df['Region'].isin(selected_regions)) &
    (df['Donor Type'].isin(selected_donor_types))
]

# Main content
st.title("🌍 AMREF Donor & Sentiment Analyzer")
st.markdown("""
This dashboard provides insights into donor contributions, sentiment analysis, and scenario planning 
for AMREF's key sectors: Primary Health Care, Climate, Education, and Livelihoods.
""")

# Key metrics
total_donations = filtered_df['Amount'].sum()
avg_donation = filtered_df['Amount'].mean()
top_donor = filtered_df.groupby('Donor')['Amount'].sum().idxmax()
top_sector = filtered_df.groupby('Sector')['Amount'].sum().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Donations", f"${total_donations:,.0f}")
col2.metric("Average Donation", f"${avg_donation:,.0f}")
col3.metric("Top Donor", top_donor)
col4.metric("Top Sector", top_sector)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Donation Analysis", 
    "Sentiment & Narrative", 
    "Scenario Planning", 
    "Data Sources"
])

with tab1:
    st.header("Donation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Donations by sector
        sector_df = filtered_df.groupby('Sector')['Amount'].sum().reset_index()
        fig = px.pie(
            sector_df, 
            values='Amount', 
            names='Sector', 
            title='Donations by Sector'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Donations by region
        region_df = filtered_df.groupby('Region')['Amount'].sum().reset_index()
        fig = px.bar(
            region_df, 
            x='Region', 
            y='Amount', 
            title='Donations by Region',
            color='Region'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top donors
    st.subheader("Top Donors")
    top_n = st.slider("Select number of top donors to display", 5, 20, 10)
    top_donors = filtered_df.groupby('Donor')['Amount'].sum().nlargest(top_n).reset_index()
    fig = px.bar(
        top_donors, 
        x='Donor', 
        y='Amount', 
        title=f'Top {top_n} Donors',
        color='Amount',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Donation trends over time
    st.subheader("Donation Trends Over Time")
    trend_df = filtered_df.groupby(['Year', 'Sector'])['Amount'].sum().reset_index()
    fig = px.line(
        trend_df, 
        x='Year', 
        y='Amount', 
        color='Sector',
        title='Donation Trends by Sector Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Social Media Sentiment & Anti-Narrative Analysis")
    
    # Synthetic sentiment data
    sentiments = {
        "Primary Health Care": {"positive": 65, "neutral": 25, "negative": 10},
        "Climate": {"positive": 55, "neutral": 30, "negative": 15},
        "Education": {"positive": 70, "neutral": 20, "negative": 10},
        "Livelihoods": {"positive": 60, "neutral": 25, "negative": 15}
    }
    
    sentiment_df = pd.DataFrame.from_dict(sentiments, orient='index').reset_index()
    sentiment_df = sentiment_df.melt(id_vars='index', var_name='Sentiment', value_name='Percentage')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment by sector
        fig = px.bar(
            sentiment_df, 
            x='index', 
            y='Percentage', 
            color='Sentiment',
            title='Social Media Sentiment by Sector',
            barmode='stack',
            color_discrete_map={
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anti-narrative detection
        st.subheader("Anti-Narrative Detection")
        anti_narratives = {
            "Primary Health Care": ["Vaccine skepticism", "Western medicine distrust"],
            "Climate": ["Climate change denial", "Anti-renewable energy"],
            "Education": ["Anti-global education", "Cultural imperialism claims"],
            "Livelihoods": ["Dependency narrative", "Local job displacement"]
        }
        
        for sector, narratives in anti_narratives.items():
            with st.expander(f"Anti-Narratives in {sector}"):
                for narrative in narratives:
                    st.write(f"- {narrative}")
                st.progress(np.random.randint(10, 50), text="Prevalence in sector")
    
    # Sentiment over time (synthetic)
    st.subheader("Sentiment Trends Over Time")
    sentiment_trend_data = []
    for sector in sentiments.keys():
        for month in range(1, 13):
            base = sentiments[sector]["positive"]
            variation = np.random.randint(-15, 15)
            positive = max(0, min(100, base + variation))
            neutral = np.random.randint(15, 35)
            negative = 100 - positive - neutral
            sentiment_trend_data.append({
                "Month": datetime(2023, month, 1).strftime("%b %Y"),
                "Sector": sector,
                "Positive": positive,
                "Neutral": neutral,
                "Negative": negative
            })
    
    sentiment_trend_df = pd.DataFrame(sentiment_trend_data)
    fig = px.line(
        sentiment_trend_df, 
        x='Month', 
        y='Positive', 
        color='Sector',
        title='Positive Sentiment Trend Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Scenario Planning & Risk Assessment")
    
    # Key sectors
    st.subheader("Key Sectors Analysis")
    sectors = ["Primary Health Care", "Climate", "Education", "Livelihoods"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Likely Scenario")
        scenario = st.selectbox(
            "Select sector for scenario analysis",
            sectors
        )
        
        if scenario:
            st.markdown(f"**{scenario} Sector Projections**")
            st.markdown("- Steady 5-7% annual growth in donations")
            st.markdown("- Moderate public support (60-70% positive sentiment)")
            st.markdown("- Low to moderate anti-narrative activity")
            st.markdown("- Stable donor base with 2-3 new major donors expected")
    
    with col2:
        st.markdown("### Worst Case Scenario")
        if scenario:
            st.markdown(f"**{scenario} Sector Risks**")
            st.markdown("- Potential 10-15% donation decline in economic downturn")
            st.markdown("- Negative sentiment spikes from health crises (for health sector)")
            st.markdown("- Increased anti-narrative activity from local groups")
            st.markdown("- Donor fatigue in traditional funding regions")
    
    # Network analysis visualization
    st.subheader("Donor Network Analysis")
    st.markdown("""
    This visualization maps donor relationships and influence patterns across AMREF's key sectors.
    """)
    
    # Legend and explanation
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Graph Legend")
        st.markdown("""
        **Bubble Color:**
        - 🔵 **Blue**: Public/Government Donors
        - 🟢 **Green**: Private/Foundation Donors  
        - 🟠 **Orange**: Multilateral Organizations
        
        **Bubble Size:**
        - Total donation amount (larger = more funding)
        
        **X-Axis (Collaboration Level):**
        - Left: Low inter-sector collaboration
        - Right: High inter-sector collaboration
        
        **Y-Axis (Funding Stability):**
        - Top: High funding stability/predictability
        - Bottom: Lower funding stability/volatility
        
        **Lines:** Partnership strength between donors
        """)
    
    with col1:
        # Generate enhanced network data with meaningful positioning
        nodes = list(filtered_df['Donor'].unique())[:15]  # Limit to 15 for visualization
        
        # Get donor lists for categorization
        df_sample = load_data()
        
        # Create meaningful data for each donor
        network_data = []
        for donor in nodes:
            donor_data = filtered_df[filtered_df['Donor'] == donor]
            total_amount = donor_data['Amount'].sum()
            sector_diversity = len(donor_data['Sector'].unique())  # Collaboration level
            
            # Determine donor type for color (simplified categorization)
            if any(keyword in donor.lower() for keyword in ['foundation', 'trust', 'fund']):
                donor_type = "Private"
                color = "green"
            elif any(keyword in donor.lower() for keyword in ['un', 'world bank', 'imf', 'unicef', 'who']):
                donor_type = "Multilateral"
                color = "orange"
            else:
                donor_type = "Public"
                color = "blue"
            
            # Position based on characteristics
            x_pos = sector_diversity + np.random.normal(0, 0.3)  # Collaboration level
            y_pos = np.log10(total_amount) + np.random.normal(0, 0.2)  # Funding stability (log scale)
            
            network_data.append({
                "Donor": donor,
                "Total_Amount": total_amount,
                "Sector_Diversity": sector_diversity,
                "Donor_Type": donor_type,
                "Color": color,
                "X_Position": x_pos,
                "Y_Position": y_pos,
                "Size": total_amount / 10000  # Scale for bubble size
            })
        
        nodes_df = pd.DataFrame(network_data)
        
        # Create the scatter plot
        fig = px.scatter(
            nodes_df,
            x="X_Position",
            y="Y_Position",
            size="Size",
            color="Donor_Type",
            hover_name="Donor",
            hover_data={
                "Total_Amount": ":$,.0f",
                "Sector_Diversity": True,
                "X_Position": False,
                "Y_Position": False,
                "Size": False
            },
            title="Donor Network Analysis: Collaboration vs Funding Stability",
            color_discrete_map={
                "Public": "blue",
                "Private": "green", 
                "Multilateral": "orange"
            },
            labels={
                "X_Position": "Collaboration Level (Inter-sector partnerships)",
                "Y_Position": "Funding Stability (Log scale of total contributions)"
            }
        )
        
        # Add connection lines between similar donors
        for i in range(len(nodes_df)):
            for j in range(i+1, len(nodes_df)):
                donor_i = nodes_df.iloc[i]
                donor_j = nodes_df.iloc[j]
                
                # Connect donors of same type or similar funding levels
                if (donor_i['Donor_Type'] == donor_j['Donor_Type'] or 
                    abs(donor_i['Y_Position'] - donor_j['Y_Position']) < 0.5):
                    
                    # Random connection strength
                    if np.random.random() > 0.7:  # Only show 30% of potential connections
                        line_width = min(3, abs(donor_i['Size'] - donor_j['Size']) / 50000 + 1)
                        fig.add_shape(
                            type="line",
                            x0=donor_i['X_Position'], y0=donor_i['Y_Position'],
                            x1=donor_j['X_Position'], y1=donor_j['Y_Position'],
                            line=dict(width=line_width, color="rgba(128,128,128,0.3)")
                        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Collaboration Level →",
            yaxis_title="Funding Stability →",
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Open Access Data Sources")
    
    data_sources = [
        {
            "Name": "OECD Creditor Reporting System (CRS)",
            "Link": "https://stats.oecd.org/Index.aspx?DataSetCode=CRS1",
            "Description": "Provides data on official development assistance (ODA), including disbursements by donor, sector, and country."
        },
        {
            "Name": "International Aid Transparency Initiative (IATI) Registry",
            "Link": "https://iatistandard.org/en/",
            "Description": "Aggregates aid and development finance data from hundreds of organizations."
        },
        {
            "Name": "World Bank Open Data",
            "Link": "https://data.worldbank.org/",
            "Description": "Includes financial flows to countries, development indicators, and lending information."
        },
        {
            "Name": "AidData",
            "Link": "https://www.aiddata.org/",
            "Description": "Offers detailed data on development finance from bilateral and multilateral donors."
        },
        {
            "Name": "UN OCHA Financial Tracking Service (FTS)",
            "Link": "https://fts.unocha.org/",
            "Description": "Tracks humanitarian aid flows in real-time."
        },
        {
            "Name": "Global Health Expenditure Database (WHO)",
            "Link": "https://apps.who.int/nha/database/Select/Indicators/en",
            "Description": "Tracks global and national health expenditures."
        },
        {
            "Name": "Global Partnership for Education (GPE) Data & Results",
            "Link": "https://www.globalpartnership.org/data-and-results",
            "Description": "Information on funding and impact in education across partner countries."
        },
        {
            "Name": "UNICEF Transparency Portal",
            "Link": "https://open.unicef.org/",
            "Description": "Real-time funding and programmatic expenditure data."
        },
        {
            "Name": "UNDP Open Data",
            "Link": "https://open.undp.org/",
            "Description": "UNDP projects and financial data globally."
        },
        {
            "Name": "Humanitarian Data Exchange (HDX)",
            "Link": "https://data.humdata.org/",
            "Description": "Includes a variety of datasets on aid, population needs, partners, and crises."
        },
        {
            "Name": "Candid (Foundation Center + GuideStar) — Open Data",
            "Link": "https://candid.org/data",
            "Description": "U.S. foundation funding trends, searchable via Foundation Maps (some open data available)."
        },
        {
            "Name": "Donor Tracker by SEEK Development",
            "Link": "https://donortracker.org/",
            "Description": "Provides up-to-date profiles on 14 major donors."
        }
    ]
    
    for source in data_sources:
        with st.expander(source["Name"]):
            st.markdown(f"**Link:** [{source['Link']}]({source['Link']})")
            st.markdown(f"**Description:** {source['Description']}")

# Footer
st.markdown("---")
st.markdown("""
**AMREF Donor & Sentiment Analyzer**  
*Last updated: June 2023*  
*Data sources: OECD, IATI, World Bank, and other open access resources*
""")