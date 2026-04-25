Predicting whether a community is on track (Good) or at risk (Bad) based on mid-month lead pacing.

Model Choice: from sklearn.naive_bayes import GaussianNB
Rule:
If difference >= 0 → Good
If difference >= -10 and difference <= -1 → Watch
If difference >= -20 and difference <= -11 → Bad
If difference <= -21 → Urgent

Digital Marketing Formulas: 
Projected EOM Leads = Current Leads ÷ Day of Month × Total Days in Month
Difference = Projected EOM Leads − Previous Month Leads
MoM % Change = Difference ÷ Previous Month Leads × 100
