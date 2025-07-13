import math
import statistics

import numpy as np
import scipy
import transformers
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5 import uic  # For loading the UI dynamically
import sys

from sympy.printing.pretty.pretty_symbology import line_width
from transformers import pipeline
import matplotlib.pyplot as plt
import scipy.stats
import scipy.stats as stats


section_agora_ans=[]
section_social_ans=[]
section_gad_ans=[]
sec1_scores=[]
sec2_scores=[]
sec3_scores=[]
weight_1=5
weight_2=3
weight_3=1
sect_number_questions=8
agora_mean = (weight_1*sect_number_questions)/((weight_1*8)+(weight_2*8)+(weight_3*8))
gad_mean = (weight_2*sect_number_questions)/((weight_1*8)+(weight_2*8)+(weight_3*8))
social_mean = (weight_3*sect_number_questions)/((weight_1*8)+(weight_2*8)+(weight_3*8))
ag_scores = [5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
gad_scores = [3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
social_scores = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
agora_stdv =statistics.stdev(ag_scores)
gad_stdv = statistics.stdev(gad_scores)
social_stdv = statistics.stdev(social_scores)
total_questions = 24


#classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
#print(len(classifier("I love this!")))
#print(classifier("I love this!").get(2))

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
#print(classifier("I love this!")['score'])
'''
for emotion in classifier("I love this"):
    if emotion['label'] == 'joy':
        joy_score = emotion['score']
        break
print(joy_score)
#print(classifier("I love this!")[3])
'''

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)

        self.sec1submitbtn.clicked.connect(self.getSec1Answs)
        self.sec2submitbtn.clicked.connect(self.getSec2Answs)
        self.sec3submitbtn.clicked.connect(self.getSec3Answs)
        self.generate_analysis.clicked.connect(self.analysis)

    def getSec1Answs(self):
        section_agora_ans.clear()
        section_agora_ans.append(self.sec1answ1.text())
        section_agora_ans.append(self.sec1answ2.text())
        section_agora_ans.append(self.sec1answ3.text())
        section_agora_ans.append(self.sec1answ4.text())
        section_agora_ans.append(self.sec1answ5.text())
        section_agora_ans.append(self.sec1answ6.text())
        section_agora_ans.append(self.sec1answ7.text())
        section_agora_ans.append(self.sec1answ8.text())
        print("section1Finished")

    def getSec2Answs(self):
        section_gad_ans.clear()
        section_gad_ans.append(self.sec2answ1.text())
        section_gad_ans.append(self.sec2answ2.text())
        section_gad_ans.append(self.sec2answ3.text())
        section_gad_ans.append(self.sec2answ4.text())
        section_gad_ans.append(self.sec2answ5.text())
        section_gad_ans.append(self.sec2answ6.text())
        section_gad_ans.append(self.sec2answ7.text())
        section_gad_ans.append(self.sec2answ8.text())
        print("section2Finished")

    def getSec3Answs(self):
        section_social_ans.clear()
        section_social_ans.append(self.sec3answ1.text())
        section_social_ans.append(self.sec3answ2.text())
        section_social_ans.append(self.sec3answ3.text())
        section_social_ans.append(self.sec3answ4.text())
        section_social_ans.append(self.sec3answ5.text())
        section_social_ans.append(self.sec3answ6.text())
        section_social_ans.append(self.sec3answ7.text())
        section_social_ans.append(self.sec3answ8.text())
        print("section3Finished")

    def getFearScores(self, ans_list,weight):
        scores=[]
        for ans in ans_list:
            print(ans)
            result = classifier(ans)
            print(result)
            for emotion in result[0]:
                print("in here")
                print(emotion)
                if emotion['label']=='fear':
                    print("found fear output")
                    scores.append(emotion['score']*weight)
                    print(emotion['score']*weight)
        return scores

    def list_sum(self,list):
        sum =0
        for i in list:
            sum+=i
        return sum

    def scoresSectionsAvg(self,l1,l2,l3,number_outcomes):
        sum_1 = self.list_sum(l1)
        sum_2 = self.list_sum(l2)
        sum_3 = self.list_sum(l3)
        sum_total = sum_1+sum_2+sum_3
        return sum_total/number_outcomes


    def fitDistData(self, data, mean, stdv):
        print(stats.t.fit(data))
        print(f"Mean: {mean} Stdv: {stdv}")

        x_bounds = np.linspace(mean-(3*stdv), mean+(3*stdv))
        t_pdf = stats.t.pdf(x_bounds, mean, stdv)
        plt.figure()
        plt.plot(x_bounds,t_pdf, 'b-', linewidth = 2, label=  "User ans distribution", )
        plt.axvline(mean)
        plt.xlabel("Scores")
        plt.ylabel("Density")
        plt.title("User Scores Distribution(t)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def weightedDist(self, data, mean, stdv):
        x_bounds = np.linespace(-3,10)
        t_pdf = stats.t.pdf(x_bounds, mean, stdv)
        n_agora_pdf = stats.norm.pdf(x_bounds, agora_mean, agora_stdv)
        n_gad_pdf = stats.norm.pdf(x_bounds, gad_mean, gad_stdv)
        n_social_pdf = stats.norm.pdf(x_bounds, social_mean, social_stdv)
        plt.figure()
        plt.plot(x_bounds, t_pdf, linewidth = 2, label="Weighted User Scores")
        plt.plot(x_bounds,n_agora_pdf, linewidth = 2, label="Aghoraphobia Range(approx)")
        plt.plot(x_bounds,n_gad_pdf, linewidth = 2, label="Social Phobia Range(approx)")
        plt.plot(x_bounds,n_social_pdf, linewidth = 2, label="W User Scores(approx)")
        plt.xlabel("Weighted Scored")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()



    def tTest(self, s_mean, p_mean, s_error, s_size):
        t_sdv = s_error/math.sqrt(s_size)
        t_score = (s_mean-p_mean)/t_sdv
        t_score = abs(t_score)
        p_val = stats.t._cdf(t_score,s_size-1)
        p_val = 2*(1-p_val)
        if p_val<0.15 :
            return ["Reject Ho",f"P-val= {p_val}"]
        else: return ["Fail to reject Ho",f"P-val= {p_val}"]


    def analysis(self):
        if len(section_agora_ans)==8 and len(section_social_ans)==8 and len(section_gad_ans)==8:
            print("All question answered...analysis started")
            sec1_scores=self.getFearScores(section_agora_ans,1)
            sec2_scores = self.getFearScores(section_gad_ans,1)
            sec3_scores = self.getFearScores(section_social_ans,1)
            scores_avg = self.scoresSectionsAvg(sec1_scores,sec2_scores,sec3_scores,(len(sec1_scores)+len(sec2_scores)+len(sec3_scores)))
            scores_total= sec1_scores+sec2_scores+sec3_scores
            self.fitDistData(scores_total, scores_avg, statistics.stdev(scores_total))

            sec_1_scores_weighted = self.getFearScores(section_agora_ans,weight_1)
            sec_2_scores_weighted = self.getFearScores(section_agora_ans, weight_2)
            sec_3_scores_weighted = self.getFearScores(section_agora_ans, weight_3)
            scores_weighted_avg = self.scoresSectionsAvg(sec_1_scores_weighted,sec_2_scores_weighted,sec_3_scores_weighted,((weight_1*8)+(weight_2*8)+(weight_3*8)))
            print(f"User Weighted Mu: {scores_weighted_avg}")
            scores_weighted_total = sec_1_scores_weighted+sec_2_scores_weighted+sec_3_scores_weighted
            print(f"User Weighted STDV: {statistics.stdev(scores_weighted_total)}")
            print(f"Agoraphobia Mu: {agora_mean} Stdv: {agora_stdv}\nSocialphobia Mu: {social_mean} Stdv: {social_stdv}\nGAD Mu: {gad_mean} Stdv: {gad_stdv}")

            self.fitDistData(scores_weighted_total, scores_weighted_avg, statistics.stdev(scores_weighted_total))

            user_results=[]
            print(self.tTest(
                scores_weighted_avg,agora_mean,statistics.stdev(scores_weighted_total),24
            ))
            user_results.append(self.tTest(scores_weighted_avg,agora_mean,statistics.stdev(scores_weighted_total),24)[0])

            print(self.tTest(
                scores_weighted_avg, gad_mean, statistics.stdev(scores_weighted_total), 24
            ))
            user_results.append(self.tTest(scores_weighted_avg, gad_mean, statistics.stdev(scores_weighted_total), 24 )[0])

            print(self.tTest(
                scores_weighted_avg, social_mean, statistics.stdev(scores_weighted_total), 24
            ))
            user_results.append(self.tTest(scores_weighted_avg, social_mean, statistics.stdev(scores_weighted_total), 24)[0])




        else:
            print("Please answer all questions")
        #print(classifier_emotion(section_agora_ans[0])[1])
        #print(classifier_emotion(section_agora_ans[0])[3])





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
