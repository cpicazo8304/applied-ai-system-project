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
            st.session_state.show_success = True
            st.rerun()


def user_profile_page():
    if st.session_state.pop("show_success", False):
        st.success("Profile created! Here are your recommendations.")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"Welcome back, {st.session_state.name}!")
    with col2:
        if st.button("🔄 Reset"):
            for key in ["user", "name", "recommendations"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.subheader("Your music profile")
    st.write("Favorite Genres:", ", ".join(st.session_state.user.favorite_genres))
    st.write("Favorite Artists:", ", ".join(st.session_state.user.favorite_artists))

    with st.expander("View your audio preferences"):
        prefs = st.session_state.user.structure_profile()
        display = {
            k.replace("preferred_", "").capitalize(): round(v, 3) if isinstance(v, float) else v
            for k, v in prefs.items()
            if k not in ("preferred_genres", "favorite_artists", "favorite_names")
        }
        st.table(display)

    st.write("Your personalized recommendations will appear here based on your profile and feedback.")
    # Only run if recommendations list is empty or a refresh is requested
    if not st.session_state.recommendations or st.session_state.get("refresh", False):
        with st.spinner("Generating recommendations..."):
            st.session_state.recommendations = st.session_state.recommender.recommend_songs(st.session_state.user, st.session_state.songs, k=5)
            st.session_state.user.update_ranked_songs(st.session_state.recommendations)
        st.session_state.refresh = False

    for idx, recommendation in enumerate(st.session_state.recommendations):
        song, _, explanation = recommendation
        st.markdown(f"**{idx+1}. {song['name']}** by {', '.join(song['artists']) if isinstance(song['artists'], list) else song['artists']} (Genre: {song['genre']})")
        st.markdown(f"_Why this song?_\n{explanation}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Like", key=f"like_{idx}"):
                st.session_state.user.like(song)
                st.session_state.refresh = True
                st.rerun()
        with col2:
            if st.button("👎 Skip", key=f"skip_{idx}"):
                st.session_state.user.skip(song)
                st.session_state.refresh = True
                st.rerun()
    

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