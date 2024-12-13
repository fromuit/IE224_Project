import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class LOLDashboard:
    """
    Dashboard class for League of Legends match predictions
    """
    def __init__(self) -> None:
        """Initialize dashboard configuration"""
        st.set_page_config(
            page_title="LoL Match Predictions",
            page_icon="🎮",
            layout="wide"
        )
        self.data: pd.DataFrame = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load the League of Legends match data
        
        Returns:
            pd.DataFrame: Loaded match data
        """
        try:
            return pd.read_csv("../Data/LCK_Tournament.csv")
        except FileNotFoundError:
            st.error("Data file not found. Please check the file path.")
            return pd.DataFrame()

    def create_feature_importance_plot(self, importance_df: pd.DataFrame) -> None:
        """
        Create feature importance visualization
        
        Args:
            importance_df: DataFrame containing feature importance scores
        """
        fig = px.bar(
            importance_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Most Important Features",
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_team_performance_plot(self) -> None:
        """Create team performance visualization"""
        team_stats = self.data.groupby("teamname").agg({
            "result": "mean",
            "kills": "mean",
            "deaths": "mean"
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Win Rate",
            x=team_stats["teamname"],
            y=team_stats["result"],
            marker_color="green"
        ))
        fig.update_layout(title="Team Performance Statistics")
        st.plotly_chart(fig, use_container_width=True)

    def create_time_metrics_plot(self) -> None:
        """Create time-based metrics visualization"""
        metrics = ["golddiffat10", "golddiffat15", "golddiffat20", "golddiffat25"]
        fig = px.box(
            self.data,
            y=metrics,
            title="Gold Difference Distribution Across Game Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_team_performance_metrics(self, team_data: pd.DataFrame) -> None:
        """Create a metrics section showing key team performance indicators"""
        st.subheader("Team Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_kda = (team_data["kills"].mean() + team_data["assists"].mean()) / team_data["deaths"].mean()
            st.metric("Average KDA", f"{avg_kda:.2f}")
        
        with col2:
            win_rate = (team_data["result"] == 1).mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            avg_game_length = team_data["gamelength"].mean() / 60  # Convert to minutes
            st.metric("Avg Game Length", f"{avg_game_length:.1f} min")
        
        with col4:
            first_blood_rate = team_data["firstblood"].mean() * 100
            st.metric("First Blood Rate", f"{first_blood_rate:.1f}%")

    def create_objective_control_plot(self, team_data: pd.DataFrame) -> None:
        """Create a plot showing objective control statistics"""
        st.subheader("Objective Control")
        
        objectives = {
            "First Dragon": "firstdragon",
            "First Herald": "firstherald",
            "First Baron": "firstbaron",
            "First Tower": "firsttower"
        }
        
        objective_rates = {
            name: team_data[col].mean() * 100 
            for name, col in objectives.items()
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(objective_rates.keys()),
                y=list(objective_rates.values()),
                marker_color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
            )
        ])
        
        fig.update_layout(
            title="Objective Control Rates (%)",
            yaxis_range=[0, 100],
            yaxis_title="Secured Rate (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_time_performance_plot(self, team_data: pd.DataFrame) -> None:
        """Create plots showing team performance across different game stages"""
        st.subheader("Performance by Game Stage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gold difference at different timestamps
            gold_diff_data = pd.DataFrame({
                "10 min": team_data["golddiffat10"],
                "15 min": team_data["golddiffat15"],
                "20 min": team_data["golddiffat20"]
            })
            
            fig = px.box(
                gold_diff_data,
                title="Gold Difference Distribution",
                labels={"value": "Gold Difference", "variable": "Game Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Kill participation over time
            kills_data = pd.DataFrame({
                "10 min": team_data["killsat10"],
                "15 min": team_data["killsat15"],
                "20 min": team_data["killsat20"]
            })
            
            fig = px.line(
                kills_data.mean(),
                title="Average Kills by Game Stage",
                labels={"value": "Average Kills", "index": "Game Time"}
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_dashboard(self) -> None:
        """Render the complete dashboard"""
        st.title("LoL Match Analysis Dashboard")
        
        # Filter controls in two columns
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            team_filter = st.selectbox(
                "Select Team",
                self.data["teamname"].unique(),
                key="team_filter"
            )
        
        with filter_col2:
            # Add "All Splits" option to the splits list
            splits = ["All Splits"] + list(self.data["split"].unique())
            split_filter = st.selectbox(
                "Select Split",
                splits,
                key="split_filter"
            )
        
        # Apply filters based on selection
        if split_filter == "All Splits":
            filtered_data = self.data[self.data["teamname"] == team_filter]
        else:
            filtered_data = self.data[
                (self.data["teamname"] == team_filter) & 
                (self.data["split"] == split_filter)
            ]
        
        # Show number of matches in filtered data
        st.caption(f"Showing data from {len(filtered_data)} matches")
        
        if len(filtered_data) == 0:
            st.warning("No matches found for this combination of team and split.")
            return
        
        # Performance metrics
        self.create_team_performance_metrics(filtered_data)
        
        # Objective control
        self.create_objective_control_plot(filtered_data)
        
        # Game length distribution and time performance
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Game Time Distribution")
            fig = px.histogram(
                filtered_data,
                x="gamelength",
                title="Game Length Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based performance
        self.create_time_performance_plot(filtered_data)
def main() -> None:
    """Main function to run the dashboard"""
    dashboard = LOLDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()