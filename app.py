import streamlit as st
import pickle

language = ['Arabic','Danish' ,'Dutch', 'English' ,'French', 'German', 'Greek' ,'Hindi',
 'Italian', 'Kannada', 'Malayalam', 'Portugeese' ,'Russian', 'Spanish',
 'Sweedish' ,'Tamil', 'Turkish']

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vector.pkl', 'rb'))

st.set_page_config(page_title="Language Detector", page_icon="📄", layout="centered", initial_sidebar_state="collapsed")

st.title("📄Language Detector")

st.write()
st.write("This is a Language Detector web app to predict the language of the text you enter in the textbox below.\n\
         The model is trained on 17 different languages and has an accuracy of 97.5%.")
st.write("The languages are: Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portugeese, Russian, Spanish, Sweedish, Tamil, Turkish")

text = st.text_input("Enter text:")

submit = st.button("Detect")
if submit:
    x = cv.transform([text]).toarray()
    ans = model.predict(x)
    st.write("The language of the text is: ", language[ans[0]])