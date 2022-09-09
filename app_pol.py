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

	st.set_page_config(page_title="predictSP")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	with overview:
		st.title("Wykorzystanie AI w komunikacji medycznej")
		st.subheader("Wspomaganie procesu rekrutacyjnego SP")
		st.subheader("AMEE Conference, Lyon 2022")
		st.write("PhD Wojciech Oronowicz-Jaśkowiak, Polish-Japanese Academy of Information Technology")
		st.write("PhD Antonina Doroszewska, Medical University of Warsaw")
		st.header("predictSP")
		st.write("(zdecydowanie nie – zdecydowanie tak)")


	with left:
		P1_slider = st.slider("Stopień w jaki SP dostosowuje swoją komunikację niewerbalną do sytuacji", value=1, min_value=1, max_value=7)
		P2_slider = st.slider("Stopień w jaki zrozumiały i poprawny gramatycznie jest tok wypowiedzi SP", value=1, min_value=1, max_value=7)

	with right:
		P4_slider = st.slider("Stopień w jakim SP okazuje emocje związane z rozmową", value=1, min_value=1, max_value=7)
		P5_slider = st.slider("Stopień w jakim SP reaguje adekwatnie do zachowania studenta", value=1, min_value=1, max_value=7)
		P6_slider = st.slider("Jeżeli pytasz się SP o informację zwrotną, to oceń sposób w jaki SP udziela informacji zwrotnej po zakończeniu rozmowy", value=3, min_value=1, max_value=7)

	data = [[P1_slider, P2_slider, P4_slider, P5_slider, P6_slider]]
	good = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Wynik pozytywny? {0}".format("Yes" if good[0] == 1 else "No"))
		st.write("Prawdopodobieństwo {0:.2f} %".format(s_confidence[0][good][0] * 100))
		st.header("Documentation (English)")
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
		st.caption("Oronowicz-Jaśkowiak, W., Doroszewska, A. (2022, August). Use of deep neural networks in evaluating medical communication. Poster presented during AMEE Conference, Lyon. International Association for Medical Education in Europe.")
		st.caption("App developed by Wojciech Oronowicz-Jaśkowiak 🤖 ML model trained by Antonina Doroszewska & Wojciech Oronowicz-Jaśkowiak 🎈 ")
		st.image("https://i.ibb.co/gRRzqST/Logo-tarcza-black-150x150-mm-0.jpg")

if __name__ == "__main__":
    main()
