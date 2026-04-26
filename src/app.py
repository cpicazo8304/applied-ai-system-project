import streamlit as st
from recommender import UserProfile, Recommender, load_songs

GENRES = ["Rock", "Hip-Hop", "Classical", "Pop", "Jazz", "R&B", "Blues", "Electronic", "Country", "Folk"]

def profile_setup_page():
    st.title("🎵 AuraTrack")
    st.subheader("Set up your music profile")

    name = st.text_input("Your name")
    favorite_genres = st.multiselect("Favorite genres", options=GENRES)
    favorite_artists = st.text_input("Favorite artists (comma separated)")

    if st.button("Create Profile"):
        if not name or not favorite_genres or not favorite_artists:
            st.error("Please fill in all fields.")
        else:
            artists_list = [a.strip() for a in favorite_artists.split(",")]
            with st.spinner("Building your profile..."):
                st.session_state.user = UserProfile(
                    favorite_genres=favorite_genres,
                    favorite_artists=artists_list
                )
                st.session_state.name = name
            st.rerun()


def user_profile_page():
    st.title(f"Welcome back, {st.session_state.name}!")
    st.subheader("Your music profile")
    st.write("Favorite Genres:", ", ".join(st.session_state.user.favorite_genres))
    st.write("Favorite Artists:", ", ".join(st.session_state.user.favorite_artists))

# initialize session state on first load
if "user" not in st.session_state:
    st.session_state.user = None
if "recommender" not in st.session_state:
    songs = load_songs("../data/songs_with_clusters.csv")
    st.session_state.recommender = Recommender(songs)
    st.session_state.songs = songs
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []


if st.session_state.user is None:
    profile_setup_page()
else:
    user_profile_page()