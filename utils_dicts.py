numerical_features = [
    'Age at Injury','Average Weekly Wage','Birth Year','IME-4 Count','Number of Dependents',
    'Days_to_C2','Days_to_C3','Days_to_First_Hearing',
    'C-2 Date_Day','C-2 Date_Month','C-2 Date_Year','C-2 Date_DayOfWeek',
    'C-3 Date_Month','C-3 Date_Day','C-3 Date_Year','C-3 Date_DayOfWeek',
    'First Hearing Date_Day','First Hearing Date_Month','First Hearing Date_Year','First Hearing Date_DayOfWeek',
    'Accident Date_Day','Accident Date_Month','Accident Date_Year','Accident Date_DayOfWeek',
    'Assembly Date_Day','Assembly Date_Month','Assembly Date_Year','Assembly Date_DayOfWeek'
]
categorical_features = [
    'Alternative Dispute Resolution_U','Alternative Dispute Resolution_Y',
    'Enc County of Injury', 'Enc District Name','Enc Industry Code',
    'Known Accident Date','Known Assembly Date','Known C-2 Date','Known C-3 Date',
    'Enc WCIO Cause of Injury Code',
    'Enc WCIO Nature of Injury Code',
    'Enc WCIO Part Of Body Code',
    'Enc Zip Code',
    'Attorney/Representative_Y',
    'Carrier Type_2A. SIF','Carrier Type_3A. SELF PUBLIC','Carrier Type_4A. SELF PRIVATE','Carrier Type_5D. SPECIAL FUND - UNKNOWN','Carrier Type_UNKNOWN',
    'COVID-19 Indicator_Y',
    'Medical Fee Region_II','Medical Fee Region_III','Medical Fee Region_IV','Medical Fee Region_UK',
    'Known First Hearing Date','Known Age at Injury','Known Birth Year',
    'Holiday_Accident','Weekend_Accident', 'Risk_Level',
    'Gender_M','Gender_Unknown',
    'Accident_Season_Sin','Accident_Season_Cos',
    
    'Relative_Wage',
    'Financial Impact Category',
    'Age_Group'
]

essential_features = [
    'IME-4 Count','Days_to_First_Hearing',
    'C-2 Date_Year','Accident Date_Year','Assembly Date_Year',
    'Attorney/Representative_Y',
    'Enc WCIO Nature of Injury Code',
    'Relative_Wage'
]

reduced_features = essential_features + [
    "Age at Injury", "Days_to_C2", "Days_to_C3", "C-2 Date_Day", "C-2 Date_DayOfWeek", 
    "C-3 Date_Day", "C-3 Date_Month", "C-3 Date_Year", "First Hearing Date_Day", "First Hearing Date_Month", "First Hearing Date_Year",
    "First Hearing Date_DayOfWeek", "Accident Date_Month"
]
