
import pandas as pd
import joblib
def load_data(file_path):
    data=pd.read_csv(file_path+"/"+'data2_for_app.csv')
    dataframe=pd.read_csv(file_path+"/"+'dataframe_for_app.csv')
    return data,dataframe
    
    
    
def load_model(file_path):
    tfv=joblib.load(file_path+"/"+'tfv_vec.pkl')
    sig=joblib.load(file_path+"/"+'sig_kernel.pkl')
    return tfv,sig

data,dataframe=load_data("C:\\Users\\nosha\\OneDrive\\Desktop\\recommendation")
tvf,sig=load_model("C:\\Users\\nosha\\OneDrive\\Desktop\\recommendation")
           


def give_recommendation(target_movie,model,data,dataframe):
# target_movie='X-Men: Days of Future Past'
    target_ind=data[data.original_title==target_movie].index.tolist()
    sigam_mat=pd.DataFrame(model)
    index10top=sigam_mat.sort_values(by=target_ind,ascending=False).head(10)
    index_list=index10top.index.tolist()
    dataframe=data.loc[index_list].original_title
    return dataframe


import streamlit as st

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to the ones you love! Just select a title and get tailored suggestions instantly.")

movie_list = data['original_title'].sort_values().to_list()

selected_movie = st.selectbox("🎞 Choose a movie you like:", movie_list)

if st.button("🎯 Get Recommendations"):
    if selected_movie:
        recommendations = give_recommendation(selected_movie, sig, data, dataframe)
        
        st.subheader(f"🎉 Movies similar to **{selected_movie}**:")
        for index, movie in enumerate(recommendations):
            st.write(f"{index + 1}. {movie}")


st.markdown("---")     
    

