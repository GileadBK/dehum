import pandas as pd  
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import numpy as np

st.set_page_config(page_title="Dehumidifier Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #54565B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold; 
        color: #54565B;
        margin: 1rem 0;
        border-bottom: 2px solid #C5203F;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #C5203F;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Dehumidifier Dashboard</div>', unsafe_allow_html=True)

csv_dir = 'csvs'
CSV_FILE = 'dehums.csv'
EXCLUDE_COLS = ["Year", "Month", "Week", "Day", "Time", "Date"]

@st.cache_data
def load_data():
    file_path = os.path.join(csv_dir, CSV_FILE)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    return df.drop_duplicates()

def get_Dehumidifier_meter_columns(df):
    all_excludes = EXCLUDE_COLS
    return [col for col in df.columns if col not in all_excludes and pd.api.types.is_numeric_dtype(df[col])]


def calculate_spikes(df, meter_cols):
    # Use DateTime if available, else fallback to Date+Time or just Time
    if not meter_cols or df.empty:
        return [], []
    df = df.copy()
    if "DateTime" in df.columns:
        time_col = "DateTime"
    elif "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
        time_col = "DateTime"
    elif "Time" in df.columns:
        time_col = "Time"
    else:
        time_col = df.columns[0] 

    # Calculate total usage per row
    time_totals = df[meter_cols].sum(axis=1, numeric_only=True)
    spike_threshold = time_totals.quantile(0.99)
    spikes = df.loc[time_totals > spike_threshold, time_col].dropna().tolist()
    inactive = df.loc[time_totals == 0, time_col].dropna().tolist()
    return spikes, inactive

def calculate_advanced_metrics(df, meter_cols):
    """Calculate advanced performance metrics"""
    metrics = {}
    
    if not meter_cols or df.empty:
        return metrics
    
    # Calculate efficiency metrics
    total_usage = df[meter_cols].sum().sum()
    avg_usage = df[meter_cols].mean().mean()
    peak_usage = df[meter_cols].max().max()
    
    # Load factor (average / peak)
    load_factor = (avg_usage / peak_usage * 100) if peak_usage > 0 else 0
    
    # Variability coefficient
    std_usage = df[meter_cols].std().mean()
    variability = (std_usage / avg_usage * 100) if avg_usage > 0 else 0
    
    metrics.update({
        'load_factor': load_factor,
        'variability': variability,
        'total_usage': total_usage,
        'avg_usage': avg_usage,
        'peak_usage': peak_usage
    })
    
    return metrics

def create_correlation_matrix(df, meter_cols):
    """Create correlation matrix heatmap"""
    if len(meter_cols) < 2:
        return None
    
    corr_matrix = df[meter_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Dehum Meter Correlation Matrix"
    )
    fig.update_layout(
        title_x=0.5,
        height=500,
        template="plotly_white"
    )
    return fig

def create_heatmap_calendar(df, meter_cols):
    """Create calendar heatmap of daily usage"""
    if df.empty or not meter_cols:
        return None
    
    df_daily = df.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['DayOfWeek'] = df_daily['Date'].dt.day_name()
    df_daily['Week'] = df_daily['Date'].dt.isocalendar().week
    
    daily_usage = df_daily.groupby(['Date', 'DayOfWeek', 'Week'])[meter_cols].sum().sum(axis=1).reset_index()
    daily_usage.columns = ['Date', 'DayOfWeek', 'Week', 'TotalUsage']
    
    pivot_data = daily_usage.pivot_table(
        values='TotalUsage', 
        index='DayOfWeek', 
        columns='Week', 
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Week", y="Day of Week", color="Dehum Usage"),
        color_continuous_scale="Viridis",
        title="Daily Dehum Usage Pattern (Calendar Heatmap)"
    )
    fig.update_layout(
        title_x=0.5,
        height=400,
        template="plotly_white"
    )
    return fig

def create_box_plot_analysis(df, meter_cols):
    """Create box plots for usage distribution analysis"""
    if df.empty or not meter_cols:
        return None
    
    df_melted = df[meter_cols + ['Month']].melt(
        id_vars=['Month'],
        value_vars=meter_cols,
        var_name='Dehum_Meter',
        value_name='Usage'
    )
    
    fig = px.box(
        df_melted,
        x='Month',
        y='Usage',
        color='Dehum_Meter',
        title="Dehum Usage Distribution by Month"
    )
    fig.update_layout(
        title_x=0.5,
        template="plotly_white",
        height=500
    )
    return fig

def apply_filters(df, date_range=None, selected_years=None, selected_months=None, selected_meters=None):
    """Apply consistent filters to any dataframe"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        try:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df_dates = pd.to_datetime(filtered["Date"], errors='coerce')
            
            # Filter by date range
            date_mask = (df_dates >= start_date) & (df_dates <= end_date)
            filtered = filtered[date_mask]
        except Exception as e:
            st.warning(f"Date filtering error: {e}")
    
    # Apply year filter
    if selected_years and "All" not in selected_years and "Year" in filtered.columns:
        if not filtered.empty:
            # Convert years to strings for consistent comparison
            filtered_years = filtered['Year'].astype(str)
            selected_years_str = [str(y) for y in selected_years]
            filtered = filtered[filtered_years.isin(selected_years_str)]
    
    # Apply month filter
    if selected_months and "All" not in selected_months and "Month" in filtered.columns:
        if not filtered.empty:
            filtered = filtered[filtered['Month'].isin(selected_months)]
    
    # Apply meter filter (keep essential columns but only analyze selected meters)
    if selected_meters and "All" not in selected_meters:
        available_meters = get_Dehumidifier_meter_columns(filtered) if not filtered.empty else []
        selected_meter_cols = [col for col in selected_meters if col in available_meters]
        
        # Keep essential columns plus selected meters
        essential_cols = ["Year", "Month", "Week", "Day", "Time", "Date"]
        cols_to_keep = []
        
        # Add essential columns that exist
        for col in essential_cols:
            if col in filtered.columns:
                cols_to_keep.append(col)
        
        # Add selected meter columns
        cols_to_keep.extend(selected_meter_cols)
        
        # Always keep CDD 15.5 for regression analysis (but not for plotting)
        if "CDD 15.5" in filtered.columns:
            cols_to_keep.append("CDD 15.5")
        
        # Only filter columns if we have valid selections
        if selected_meter_cols:
            filtered = filtered[cols_to_keep]
    
    return filtered

def filter_by_sidebar(df):
    st.sidebar.markdown("## Dashboard Filters")
    
    # Initialize filters with error handling
    try:
        # Date range filter
        min_date = pd.to_datetime(df["Date"]).min()
        max_date = pd.to_datetime(df["Date"]).max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="sidebar_date_range"
        )
    except Exception:
        date_range = None
        st.sidebar.warning("Date range not available")
    
    # Filter options with validation
    try:
        unique_years = ["All"] + sorted([str(y) for y in df['Year'].unique() if pd.notna(y)])
        unique_months = ["All"] + sorted([m for m in df['Month'].unique() if pd.notna(m)])
        # FIX: Define Dehum_meter_options here
        Dehum_meter_options = ["All"] + get_Dehumidifier_meter_columns(df)
    except Exception as e:
        st.sidebar.error(f"Error loading filter options: {e}")
        unique_years = ["All"]
        unique_months = ["All"] 
        Dehum_meter_options = ["All"]

    # Sidebar selections
    selected_meters = st.sidebar.multiselect(
        "Dehum Meters", 
        options=Dehum_meter_options, 
        default=["All"],
        help="Select specific Dehum meters to analyze"
    )
    selected_years = st.sidebar.multiselect(
        "Years", 
        options=unique_years, 
        default=["All"],
        help="Filter by specific years"
    )
    selected_months = st.sidebar.multiselect(
        "Months", 
        options=unique_months, 
        default=["All"],
        help="Filter by specific months"
    )
    
    # Apply filters using the new function
    filtered = apply_filters(df, date_range, selected_years, selected_months, selected_meters)
    
    # Show filter results
    if not filtered.empty:
        st.sidebar.success(f"{len(filtered):,} records after filtering")
    else:
        st.sidebar.error("No data matches current filters")
    
    return filtered, selected_meters, selected_years, selected_months, date_range


def main():
    df = load_data()
    filtered, selected_meters, selected_years, selected_months, date_range = filter_by_sidebar(df)

    Dehumidifier_meter_columns = get_Dehumidifier_meter_columns(filtered)

    if selected_meters and "All" not in selected_meters:
        metric_cols = [col for col in selected_meters if col in Dehumidifier_meter_columns]
    else:
        metric_cols = Dehumidifier_meter_columns

    # ========== SECTION 1: KEY PERFORMANCE INDICATORS ==========
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    totalDehumFlow = highest_Dehum = average_Dehum = highest_Dehum_type = totalIncomingVsMain = main_Dehum = "N/A"
    
    if not filtered.empty and metric_cols:
        meter_totals = filtered[metric_cols].sum(numeric_only=True)
        highest_Dehum = meter_totals.max()
        highest_Dehum_type = meter_totals.idxmax()
        average_Dehum = meter_totals.mean().round(1)
        totalDehumFlow = meter_totals.sum()

    with col1:
        st.metric(
            "Highest Dehum Flow", 
            f"{highest_Dehum:,.0f}" if isinstance(highest_Dehum, (int, float)) else highest_Dehum,
            border=True
        )
    with col2:
        st.metric(
            "Total Dehum Flow", 
            f"{totalDehumFlow:,.0f}" if isinstance(totalDehumFlow, (int, float, np.number)) else totalDehumFlow,
            border=True
        )
    with col3:
        st.metric(
            "Highest Dehum Type", 
            highest_Dehum_type if isinstance(highest_Dehum_type, str) else f"{highest_Dehum_type}",
            border=True
        )
    with col4:
        st.metric(
            "Average Dehum Usage", 
            f"{average_Dehum:,.0f}" if isinstance(average_Dehum, (int, float)) else "N/A",
            border=True
        )


    # ========== SECTION 2: TIME SERIES ANALYSIS ========== 
    st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    col_ts1, col_ts2 = st.columns([3, 1])
    
    with col_ts2:
        resample_interval = st.selectbox(
            "Time Interval",
            options=[("15 Minutes", "15T"), ("1 Hour", "H"), ("1 Day", "D")],
            format_func=lambda x: x[0],
            index=0
        )[1]
        
        chart_type = st.selectbox(
            "Chart Type",
            options=["Line", "Area"],
            index=0
        )
    
    with col_ts1:
        if not filtered.empty and metric_cols:
            # Prepare time series data
            filtered_ts = filtered.copy()
            if "Date" in filtered_ts.columns and "Time" in filtered_ts.columns:
                filtered_ts["DateTime"] = pd.to_datetime(
                    filtered_ts["Date"].astype(str) + " " + filtered_ts["Time"].astype(str), 
                    errors="coerce"
                )
                filtered_ts = filtered_ts.dropna(subset=["DateTime"])
                filtered_ts = filtered_ts.set_index("DateTime").resample(resample_interval).sum(numeric_only=True).reset_index()
                x_col = "DateTime"
            else:
                x_col = "Time" if "Time" in filtered_ts.columns else filtered_ts.columns[0]
            
            fig = go.Figure()
            
            # Use metric_cols instead of Dehum_meter_columns to respect filtering
            meters_to_plot = metric_cols if metric_cols else Dehumidifier_meter_columns

            for meter in meters_to_plot:
                if meter in filtered_ts.columns:
                    if chart_type == "Area":
                        fig.add_trace(go.Scatter(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines',
                            fill='tonexty' if meter != meters_to_plot[0] else 'tozeroy',
                            name=meter,
                            stackgroup='one'
                        ))
                    elif chart_type == "Bar":
                        fig.add_trace(go.Bar(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            name=meter
                        ))
                    else:  # Line chart
                        fig.add_trace(go.Scattergl(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines+markers',
                            name=meter,
                            hovertemplate=f"<b>{meter}</b><br>Usage: %{{y}}<extra></extra>"
                        ))
            
            fig.update_layout(
                title="Dehum Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Dehum Usage",
                template="plotly_white",
                height=500,
                xaxis=dict(
                    showgrid=True,
                    rangeslider=dict(visible=True) if chart_type == "Line" else dict(visible=False)
                ),
                yaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

    col7, col8, col9 = st.columns(3)
    col10, col11, col12, col13 = st.columns(4)

    # Prepare for advanced metrics
    daytime_total = nighttime_total = 0
    day_vs_night_change = "N/A"
    delta_color_day_night = "off"
    spike_intervals_list, inactive_periods_list = [], []
    weekday_total = weekend_total = 0
    weekday_vs_weekend_change = "N/A"
    delta_color_weekday_weekend = "off"
    current_week_change = "N/A"
    delta = None
    delta_color = "off"
    safe_Dehum_meter_columns = []

    filtered_data = filtered.copy()
    combined_data = filtered.copy()

    if not filtered_data.empty:
        Dehum_meter_columns_adv = [col for col in filtered_data.columns if col not in ["Year", "Month", "Week", "Day", "Time", "Date", "DateTime"] and pd.api.types.is_numeric_dtype(filtered_data[col])]
        for col in Dehum_meter_columns_adv:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

        # Calculate spikes and inactive periods
        spike_intervals_list, inactive_periods_list = calculate_spikes(filtered_data, Dehum_meter_columns_adv)

        # --- Day vs Night Calculation ---
        if "DateTime" in filtered_data.columns:
            filtered_data["Hour"] = filtered_data["DateTime"].dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        elif "Time" in filtered_data.columns:
            filtered_data["Hour"] = pd.to_datetime(filtered_data["Time"], errors="coerce").dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        else:
            daytime = nighttime = pd.DataFrame()

        filtered_data.drop(columns=["Hour"], inplace=True, errors='ignore')

        daytime_total = daytime[Dehum_meter_columns_adv].sum().sum() if not daytime.empty else 0
        nighttime_total = nighttime[Dehum_meter_columns_adv].sum().sum() if not nighttime.empty else 0

        if (nighttime_total + daytime_total) > 0:
            day_vs_night_change = round(((daytime_total - nighttime_total) / (nighttime_total + daytime_total)) * 100, 1)
            delta_color_day_night = "normal" if day_vs_night_change >= 0 else "inverse"
        else:
            day_vs_night_change = "N/A"
            delta_color_day_night = "off"

        # --- Weekday vs Weekend Change ---
        if "Day" in combined_data.columns:
            weekday_data = combined_data[combined_data["Day"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
            weekend_data = combined_data[combined_data["Day"].isin(["Sat", "Sun"])]
            safe_Dehum_meter_columns = [col for col in Dehum_meter_columns_adv if col in weekday_data.columns]
            weekday_total = weekday_data[safe_Dehum_meter_columns].sum().sum() if not weekday_data.empty else 0
            weekend_total = weekend_data[safe_Dehum_meter_columns].sum().sum() if not weekend_data.empty else 0
            if weekend_total > 0:
                weekday_vs_weekend_change = round(((weekday_total - weekend_total) / weekend_total) * 100, 1)
                delta_color_weekday_weekend = "normal" if weekday_vs_weekend_change >= 0 else "inverse"
            else:
                weekday_vs_weekend_change = "N/A"
                delta_color_weekday_weekend = "off"

        # --- Week over week ---
        if not filtered_data.empty and "Week" in filtered_data.columns and "Year" in filtered_data.columns:
            # Get current and previous week numbers
            current_weeks = filtered_data["Week"].unique()
            if len(current_weeks) > 0:
                current_week = max(current_weeks)
                current_week_data = filtered_data[filtered_data["Week"] == current_week]
                previous_week = current_week - 1
                previous_week_data = filtered_data[filtered_data["Week"] == previous_week]
                current_week_total = current_week_data[Dehum_meter_columns_adv].sum().sum() if not current_week_data.empty else 0
                previous_week_total = previous_week_data[Dehum_meter_columns_adv].sum().sum() if not previous_week_data.empty else 0
                if previous_week_total > 0:
                    current_week_change = round(((current_week_total - previous_week_total) / previous_week_total) * 100, 1)
                    delta_color = "normal" if current_week_change >= 0 else "inverse"
                    delta = current_week_change

    with col7:
        st.metric(
            label="Day vs Night Change (%)",
            value="",
            delta=f"{day_vs_night_change}%" if day_vs_night_change != "N/A" else "None",
            delta_color=delta_color_day_night,
            label_visibility="visible",
            border=True
        )
    with col8:
        st.metric(
            label="Weekday vs Weekend Change (%)",
            value="",
            delta=f"{weekday_vs_weekend_change}%" if weekday_vs_weekend_change != "N/A" else "None",
            delta_color=delta_color_weekday_weekend,
            label_visibility="visible",
            border=True
        )
    with col9:
        st.metric(
            label="Week Over Week Change (%)",
            value="",
            delta=f"{delta}%" if delta is not None else "None",  
            delta_color= delta_color if delta is not None else "off",  
            label_visibility="visible",
            border=True
        )
    with col12:
        with st.expander("Day vs Night Dehum Usage"):
            st.write(f"Daytime Total: {daytime_total:,.0f}")
            st.write(f"Nighttime Total: {nighttime_total:,.0f}")
    with col13:
        with st.expander("Weekday vs Weekend Dehum Usage"):
            st.write(f"Weekday Total: {weekday_total:,.0f}")
            st.write(f"Weekend Total: {weekend_total:,.0f}")
    with col11:
        with st.expander("Inactive Periods"):
            if inactive_periods_list:
                for interval in inactive_periods_list:
                    st.write(f"Inactive at: {interval}")
            else:
                st.write("No inactive periods detected.")
    with col10:
        with st.expander("Spike Time Intervals"):
            if spike_intervals_list:
                for interval in spike_intervals_list:
                    st.write(f"Spike at: {interval}")
            else:
                st.write("No spike time intervals detected.")

    # ========== SECTION 3: ADVANCED ANALYTICS ==========
    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Usage Patterns", "Correlations", "Calendar View", "Distributions"])
    
    with tab1:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            if not filtered.empty and metric_cols:
                # Usage breakdown pie chart
                meter_totals = filtered[metric_cols].sum(numeric_only=True)
                usage_breakdown = pd.DataFrame({
                    "Dehum Meter": meter_totals.index,
                    "Total Usage": meter_totals.values
                })
                usage_breakdown = usage_breakdown[usage_breakdown["Total Usage"] > 0]

                if not usage_breakdown.empty:
                    fig_pie = px.pie(
                        usage_breakdown,
                        names="Dehum Meter",
                        values="Total Usage",
                        title="Dehum Usage Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No usage data available for selected meters.")
            else:
                st.info("No data available for the selected filters.")

        with col_p2:
            if not filtered.empty and metric_cols:
                # Day vs Night comparison
                filtered_copy = filtered.copy()
                if "DateTime" in filtered_copy.columns:
                    filtered_copy["Hour"] = filtered_copy["DateTime"].dt.hour
                elif "Time" in filtered_copy.columns:
                    filtered_copy["Hour"] = pd.to_datetime(filtered_copy["Time"], errors="coerce").dt.hour
                else:
                    filtered_copy["Hour"] = 12 
                
                filtered_copy["Period"] = filtered_copy["Hour"].apply(
                    lambda x: "Day (6AM-6PM)" if 6 <= x < 18 else "Night (6PM-6AM)"
                )
                
                period_usage = filtered_copy.groupby("Period")[metric_cols].sum().sum(axis=1).reset_index()
                period_usage.columns = ["Period", "Total Usage"]
                
                fig_period = px.bar(
                    period_usage,
                    x="Period",
                    y="Total Usage",
                    title="Day vs Night Usage Comparison",
                    color="Period",
                    color_discrete_sequence=["#FFA500", "#4169E1"]
                )
                fig_period.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_period, use_container_width=True)
    
    with tab2:
        if not filtered.empty and len(metric_cols) > 1:
            corr_fig = create_correlation_matrix(filtered, metric_cols)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Need at least 2 Dehum meters for correlation analysis.")
    
    with tab3:
        if not filtered.empty and metric_cols:
            calendar_fig = create_heatmap_calendar(filtered, metric_cols)
            if calendar_fig:
                st.plotly_chart(calendar_fig, use_container_width=True)
        else:
            st.info("No data available for calendar view.")
    
    with tab4:
        if not filtered.empty and metric_cols:
            box_fig = create_box_plot_analysis(filtered, metric_cols)
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("No data available for distribution analysis.")

   

if __name__ == "__main__":
    main()
