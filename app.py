import streamlit as st
import pickle
import warnings
import keras
import sklearn
import matplotlib
warnings.filterwarnings('ignore')

from datetime import datetime
startTime = datetime.now()

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

X_d = {0:"No",1:"Yes"}
def main():

	st.set_page_config(page_title="predictSP")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	with overview:
		st.title("Using artificial intelligence to evaluate medical communication")
		st.subheader("How to evaluate desired simulated patients features during recruitment process?")
		st.subheader("AMEE Conference, Lyon 2022")
		st.write("PhD Wojciech Oronowicz-Jaśkowiak, Polish-Japanese Academy of Information Technology")
		st.write("PhD Antonina Doroszewska, Medical University of Warsaw")
		st.header("predictSP")
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
			st.write('The application is designed to support the recruitment process of simulated patients. It was created on the basis of the experiences of teachers working with SP at the Department of Medical Communication of the Medical University of Warsaw, which show that SP often quit their jobs and it is necessary to constantly recruit new people. In the recruitment process, it is crucial to choose the right people who will carry out the tasks set for the SPs during the classes. The application uses deep neural network, thanks to which people participating in the recruitment, by answering questions, can obtain information whether a given candidate can perform the role of an SP. The results may support the decision-making process in selecting candidates for the SP.')
		else:
			st.write('')
		if st.button('Practical use of our app'):
			st.write('The application is worth using during recruitment interviews with candidates for SP. An important element of these interviews should be playing a short interview scenario between the patient and the doctor, which will allow for a preliminary assessment of how the candidate plays the assigned role. After the scenario has been played, the followers can independently answer the questions provided in the app. Then discuss the observations and the results of estimation made by the deep neural network.')
		else:
			st.write('')
		st.caption("We'd love to hear feedback ❤️ antonina.doroszewska@wum.edu.pl | oronowiczjaskowiak@pjwstk.edu.pl")
		st.caption("Oronowicz-Jaśkowiak, W., Doroszewska, A. (2022, August). Use of deep neural networks in evaluating medical communication. Poster presented during AMEE Conference, Lyon. International Association for Medical Education in Europe.")
		st.caption("App developed by Wojciech Oronowicz-Jaśkowiak 🤖 ML model trained by Antonina Doroszewska & Wojciech Oronowicz-Jaśkowiak 🎈 ")
		st.image("https://i.ibb.co/gRRzqST/Logo-tarcza-black-150x150-mm-0.jpg")

if __name__ == "__main__":
    main()
