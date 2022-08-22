import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
startTime = datetime.now()

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

X_d = {0:"No",1:"Yes"}
def main():

	st.set_page_config(page_title="AME2022")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	with overview:
		st.title("Using artificial intelligence to evaluate medical communication")
		st.subheader("How to evaluate desired simulated patients features during recruitment process?")
		st.subheader("AMEE Conference, Lyon 2022")
		st.write("PhD Wojciech Oronowicz-Jaśkowiak, Polish-Japanese Academy of Information Technology")
		st.write("PhD Antonina Doroszewska, Medical University of Warsaw")
		st.header("App")
		st.write("(definitely no – definitely yes)")


	with left:
		P1_slider = st.slider("Whether the simulated patient adjusts his/her non-verbal communication to the situation", value=1, min_value=1, max_value=7)
		P2_slider = st.slider("Whether the simulated patient is understandable and grammatically correct", value=1, min_value=1, max_value=7)

	with right:
		P4_slider = st.slider("Whether the simulated patient shows the emotion during the conversation/related to the conversation", value=1, min_value=1, max_value=7)
		P5_slider = st.slider("Whether the simulated patient responds adequately to the student's behavior", value=1, min_value=1, max_value=7)
		P6_slider = st.slider("If you ask the simulated patient for feedback, evaluate the way in which the SP provides feedback after the end of the class (substantively and technically)", value=3, min_value=1, max_value=7)

	data = [[P1_slider, P2_slider, P4_slider, P5_slider, P6_slider]]
	good = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Positive outcome? {0}".format("Yes" if good[0] == 1 else "No"))
		st.write("Probability {0:.2f} %".format(s_confidence[0][good][0] * 100))
		st.header("Documentation")
		if st.button('How does it work?'):
			st.write('https://bit.ly/3wk5YSh')
		else:
			st.write('')
		if st.button('How does it work? (simplyfield)'):
			st.write('...')
		else:
			st.write('')
		if st.button('Practical use of our app'):
			st.write('...')
		else:
			st.write('')
		st.caption("Oronowicz-Jaśkowiak, W., Doroszewska, A. (2022, August). Use of deep neural networks in evaluating medical communication. Poster presented during AMEE Conference, Lyon. International Association for Medical Education in Europe.")
		st.caption("The model was deployed using streamlit app 🎈 App developed by Wojciech Oronowicz-Jaśkowiak, PhD 🤖")
		st.image("https://i.ibb.co/gRRzqST/Logo-tarcza-black-150x150-mm-0.jpg")

if __name__ == "__main__":
    main()
