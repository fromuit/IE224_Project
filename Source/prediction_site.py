"""
prediction_site.py - Streamlit application for LCK Match Prediction
"""
import streamlit as st
import pandas as pd
import joblib
from predictor import DraftBasedPredictor
# Page config
st.set_page_config(
    page_title="LCK Match Predictor",
    page_icon="🎮",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    try:
        model_data = joblib.load('../Models/draft_predictor.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load champion list
@st.cache_data
def load_champion_list():
    """Load and cache the list of available champions"""
    df = pd.read_csv("../Data/processed_for_prediction.csv")
    champions = sorted(list(set(
        df['pick1'].unique().tolist() +
        df['pick2'].unique().tolist() +
        df['pick3'].unique().tolist() +
        df['pick4'].unique().tolist() +
        df['pick5'].unique().tolist()
    )))
    return champions

def main():
    st.title("Dự đoán kết quả LCK 🎮")
    st.write("Dự đoán tỉ lệ thắng dựa trên đội tuyển và lượt pick tướng")
    
    # Load data
    predictor = load_predictor()
    champions = load_champion_list()
    
    # Create two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Đội 1 (Đội xanh)")
        team1_name = st.text_input("Tên đội:", value="T1")
        st.write("Chọn tướng:")
        team1_picks = []
        for i in range(5):
            pick = st.selectbox(
                f"Pick {i+1}",
                options=champions,
                key=f"team1_pick{i}"
            )
            team1_picks.append(pick)

    with col2:
        st.subheader("Đội 2 (Đội đỏ)")
        team2_name = st.text_input("Tên đội:", value="Gen.G")
        st.write("Chọn tướng:")
        team2_picks = []
        for i in range(5):
            pick = st.selectbox(
                f"Pick {i+1}",
                options=champions,
                key=f"team2_pick{i}"
            )
            team2_picks.append(pick)

    # Predict button
    if st.button("Kết quả"):
        try:
            # Get prediction
            result = predictor.predict_match(
                team1_name=team1_name,
                team1_picks=team1_picks,
                team2_name=team2_name,
                team2_picks=team2_picks
            )
            
            # Display results
            st.header("Kết quả dự đoán")
            
            # Win probabilities
            st.subheader("Tỉ lệ thắng mỗi đội")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric(
                    team1_name,
                    f"{result['team1']['win_probability']:.1%}"
                )
            with prob_col2:
                st.metric(
                    team2_name,
                    f"{result['team2']['win_probability']:.1%}"
                )
            
            # Champion Statistics
            st.subheader("Thông số các tướng lựa chọn")
            
            # Team 1 Stats
            st.write(f"\n{team1_name} Draft:")
            for stat in result['team1']['champion_stats']:
                with st.expander(f"Lượt {stat['position']} - {stat['champion']}"):
                    st.write(f"Winrate của đội khi pick ở lượt {stat['position']}: {stat['team_winrate']:.1%} ({stat['team_games']} games)")
                    st.write(f"Winrate của đội khi pick tướng này: {stat['team_all_pick_winrate']:.1%} ({stat['team_all_pick_games']} games)")
                    st.write(f"Winrate của tướng này khi pick ở lượt {stat['position']}: {stat['overall_winrate']:.1%} ({stat['overall_games']} games)")
                    st.write(f"Winrate tổng của tướng này: {stat['overall_all_pick_winrate']:.1%} ({stat['overall_all_pick_games']} games)")
            
            # Team 2 Stats
            st.write(f"\n{team2_name} Draft:")
            for stat in result['team2']['champion_stats']:
                with st.expander(f"Lượt {stat['position']} - {stat['champion']}"):
                    st.write(f"Winrate của đội khi pick ở lượt {stat['position']}: {stat['team_winrate']:.1%} ({stat['team_games']} games)")
                    st.write(f"Winrate của đội khi pick tướng này: {stat['team_all_pick_winrate']:.1%}({stat['team_all_pick_games']} games)")
                    st.write(f"Winrate của tướng này khi pick ở lượt {stat['position']}: {stat['overall_winrate']:.1%} ({stat['overall_games']} games)")
                    st.write(f"Winrate tổng của tướng này: {stat['overall_all_pick_winrate']:.1%} ({stat['overall_all_pick_games']} games)")
            
                        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()