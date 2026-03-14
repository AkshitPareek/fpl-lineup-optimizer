
# Add to settings page or sidebar

dark_mode = st.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stDataFrame {
            background-color: #2d2d2d;
        }
    </style>
    """, unsafe_allow_html=True)
