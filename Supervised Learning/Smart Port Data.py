import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the classes for Linear and Logistic Regression
from sklearn.linear_model import LinearRegression, LogisticRegression
# Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import svm model
from sklearn import svm
# Import Naive_Bayes model
from sklearn.naive_bayes import GaussianNB
# For Chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Import preprocessing for LabelEncoder en OneHotEncoder
from sklearn import preprocessing
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy, precision and recall calculation
from sklearn import metrics

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180


port = pd.read_excel("/home/becode/Downloads/CASD-4113 Smart Shipping Studie Anomized.xlsx")

print(port.head(10))
# Observation : class = the label/target with values p(oisonous) and e(dible),
# the other 22 columns are possible features
# Observation : all the data is discrete and categorical, nominal but not ordinal
print(f"\nSome stats:\n ",port.describe())
# Observation : all the columns are discrete values => take into account for Classifier selection

print(f"\nshape dataframe: {port.shape}")
#print("\nChecking for NA values:\n",port.isna().sum())

# drop unnecessary columns : 'Verblijf' and 'Reis' can go, they are captured in the 'FD nummer ANO'
# 'Product' is captured in the more practical 'Productcode' but let's first make a table linking the two
#products = port.groupby('Productcode')['Product']
#print(products)

# DIT IS TABEL MET PRODUCTCODES VS PRODUCT
codes = port.Productcode.unique()
product = port.Product.unique()
print(codes)
print(len(codes))
print(product)
print(len(product))
#Product_list = pd.DataFrame({'Productcode': codes, 'Product':product})
#print(Product_list.head(10))
# SOME PRODUCTCODES ARE ASSIGNED WRONG

port = port.drop(columns=['Verblijf','Product'])
#print(port.head(10))


# We can drop the rows where fields in 'Haven bestemming' and 'Haven herkomst' are missing,
# they constitute only 10 rows on the total

#port = port.drop(index = , axis=0)
#rows_togo = port[port['Bestemming'].isna()
#togo = port[port[['Haven bestemming']['Haven herkomst'].isna()].index.tolist()
rows_togo = [index for index, row in port.iterrows() if row.isnull().any()]
for rowindex in rows_togo:
    port = port.drop([port.index[rowindex]])
print(port.shape)


"""
CHANGE 'MAX LENGTE' BY CLASS Nr
Klasse 2	<-	38,51 - 55 m - KEMP
Klasse 3	<-	55,01 - 85 m
Klasse 4	<-	85,01 - 105 m
Klasse 5	<-	105,01 - 135,00 m
Klasse 6-7	<-	135,01 - .. m
"""
port['Max lengte'] = port['Max lengte'].apply(lambda x: 'Class1' if x<38.5 else ('Class2' if 38.51<x<55 else ('Class3' if 55.01<x<85 else ('Class4' if 85.01<x<105 else ('Class5' if 105.01<x<135 else 'Class6_7')))))

# Adjust 'Scheepstypes' from 18 to 8 categories
ship_type = {'Liquid bulk' : ['TANKER, (GEEN GAS) (ZEE)','GAS-TANKSCHIP', 'MOTORTANKSCHIP, VL. LADING TYPE N', 'MOTORTANKSCHIP, VL. LADING TYPE C', 'BUNKERSCHIP (BINV)','MOTORTANKSCHIP'],'Dry bulk':['MOTORVRACHTSCHIP', 'MOTORTANKSCHIP, DROGE LADING', 'TANKDUWBAK (TDB), DROGE LADING'], 'Push boat (Dry bulk)':['VRACHTDUWBAK (VDB)', 'SLEEP-VRACHTSCHIP', 'DUWBOOT LOSVAREND'], 'Push boat (Liquid bulk)' :['TANKDUWBAK (TDB), DROGE LADING', 'DUWBAK, CHEMISCH'], 'Container ship' : ['CONTAINERSCHIP'], 'Offshore ship' :['WERKVAARTUIG (BINV)'], 'Other' :['PONTON'], 'Tugboat':['SLEEPBOOT LOSVAREND']}
port['Scheepstype'] = port['Scheepstype'].apply(lambda x: 'Liquid bulk' if x in ship_type['Liquid bulk'] else ('Dry bulk' if x in ship_type['Dry bulk'] else ( 'Push boat (Dry bulk)' if x in ship_type['Push boat (Dry bulk)'] else ( 'Push boat (Liquid bulk)' if x in ship_type['Push boat (Liquid bulk)'] else ( 'Container ship' if x in ship_type['Container ship'] else ( 'Offshore ship' if x in ship_type['Offshore ship'] else ('Other' if x in ship_type['Other'] else 'Tugboat')) )) )))
#print(port.Scheepstype)

# TIME FOR SOME METRICS


port2015 = port[port['Aanvang jaar'] == 2015]
#print(port2015['FD nummer ANO'].value_counts())
port2019 = port[port['Aanvang jaar'] == 2019]

#passages per uniek trip of reiscode!
print("\npassages per unique trip or 'Reis' code!")
passages_2015 = len(port2015['Reis'].unique())#port2015.groupby('Reis')['FD nummer ANO'].value_counts()
print("passages 2015 : ", passages_2015)
passages_2019 = len(port2019['Reis'].unique())#groupby('Reis')['FD nummer ANO'].value_counts()
print("passages 2019 : ", passages_2019)

#gemiddeld aantal producten per reis:
print("\ngemiddeld aantal producten per reis:")
prods = port2015.groupby('Reis')['FD nummer ANO'].value_counts()
print("2015 : Gemiddeld aantal producten per reis :", prods.mean())
prods = port2019.groupby('Reis')['FD nummer ANO'].value_counts()
print("2019 : Gemiddeld aantal producten per reis :", prods.mean())

# SHIPS AND PASSAGES
print("\nNUMBER OF SHIPS :")
print("Number of individual/different ships in 2015")
print(len(port[port['Aanvang jaar'] == 2015]['FD nummer ANO'].unique()))
print("Number of individual/different ships in 2019")
print(len(port[port['Aanvang jaar'] == 2019]['FD nummer ANO'].unique()))

# doorvaart vs vertrek/aankomst
"""
dvrt = port2015['Route categorie'][port2015['Route categorie' == 'Doorvaart']].count()
vert = port2015['Route categorie'][port2015['Route categorie' == 'Vertrek']].count()
ankmst = port2015['Route categorie'][port2015['Route categorie' == 'Aankomst']].count()
"""
#Moet ik hier ook ni dubbele velden voor verschillende producten eruit halen voor ik dit bereken??
# de percenten kloppen omdat dat verhiudingen zijn, absoluut moetik de verschillend eproductlijnen per reis eruit halen
dvrt = port2015[port2015['Route categorie'] == 'Doorvaart']['Route categorie'].count()
vert = port2015[port2015['Route categorie'] == 'Vertrek']['Route categorie'].count()
ankmst = port2015[port2015['Route categorie'] == 'Aankomst']['Route categorie'].count()
prct_dvrt = round((dvrt / (dvrt + vert + ankmst))*100,2)
prct_vert = round((vert / (dvrt + vert + ankmst))*100,2)
prct_ankmst = round((ankmst / (dvrt + vert + ankmst))*100,2)
print("\nRatios Doorvaart, Vertrek and Aankomst:")
print("2015 : %doorvaart: ", prct_dvrt," %vertrek: ",prct_vert, " %aankomst : ", prct_ankmst )

dvrt = port2019[port2019['Route categorie'] == 'Doorvaart']['Route categorie'].count()
vert = port2019[port2019['Route categorie'] == 'Vertrek']['Route categorie'].count()
ankmst = port2019[port2019['Route categorie'] == 'Aankomst']['Route categorie'].count()
prct_dvrt = round((dvrt / (dvrt + vert + ankmst))*100,2)
prct_vert = round((vert / (dvrt + vert + ankmst))*100,2)
prct_ankmst = round((ankmst / (dvrt + vert + ankmst))*100,2)
print("2019 : %doorvaart: ", prct_dvrt," %vertrek: ",prct_vert, " %aankomst : ", prct_ankmst )

# per schip unieke karakteristieken?

# TYPE OF SHIPS descending FOR DOORVAART AND STAY !!!! TYPE OF SHIPS? MOETIK PER DOORVAART OF STAY OOK NI KIJKEN NAAR SIZE??
# Here I drop the productcode that is reponsible for double lines per ship, I do this for ship metrics unrelated to product information
print("\n Type of ships, 'Max lentge' and 'Scheepstype' for Doorvaart + Vertrek/Aankomst:")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
ships_type_2015 = port2015_no_productcode['Scheepstype'].value_counts()
ships_class_2015 = port2015_no_productcode['Max lengte'].value_counts()
print("->Scheepstype for 2015 :")
print(ships_type_2015.sort_values(ascending=False))
print("->Max lengte for 2015 :")
print(ships_class_2015.sort_values(ascending=False))
ships_type_2019 = port2019_no_productcode['Scheepstype'].value_counts()
ships_class_2019 = port2019_no_productcode['Max lengte'].value_counts()
print("->Scheepstype for 2019 :")
print(ships_type_2019.sort_values(ascending=False))
print("->Max lengte for 2019 :")
print(ships_class_2019.sort_values(ascending=False))
#print("Check upper sum of ships matches passages 2015, OK," , ships_type_2015.sum())
#print("Check upper sum of ships matches passages 2019, OK," , ships_type_2019.sum())

prct_ships_type_2015 = round((ships_type_2015 / ships_type_2015.sum())*100,4)
print("->Scheepstype for 2015 in %: ")
print(prct_ships_type_2015)
prct_ships_type_2019 = round((ships_type_2019 / ships_type_2019.sum())*100,4)
print("->Scheepstype for 2019 in %: ")
print(prct_ships_type_2019)
##
#plt.bar(elements, edible, bottom=pois, color='blue', edgecolor='white',label='edible')
plt.bar(prct_ships_type_2019.iloc[:,0], prct_ships_type_2019.iloc[:,1])
plt.show()
##
prct_ships_class_2015 = round((ships_class_2015 / ships_class_2015.sum())*100,4)
print("->Max lengte for 2015 in %: ")
print(prct_ships_class_2015)
prct_ships_class_2019 = round((ships_class_2019 / ships_class_2019.sum())*100,4)
print("->Max lengte for 2019 in %: ")
print(prct_ships_class_2019)

# TYPE OF SHIPS descending FOR DOORVAART
print("\nType of ships, 'Max lengte' and 'Scheepstype', for Doorvaart: ")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
port2015_no_productcode = port2015_no_productcode[port2015_no_productcode['Route categorie'] == 'Doorvaart']
port2019_no_productcode = port2019_no_productcode[port2019_no_productcode['Route categorie'] == 'Doorvaart']
ships_type_2015 = port2015_no_productcode['Scheepstype'].value_counts()
ships_class_2015 = port2015_no_productcode['Max lengte'].value_counts()
print("->Scheepstype for 2015 :")
print(ships_type_2015.sort_values(ascending=False))
print("->Max lengte for 2015 :")
print(ships_class_2015.sort_values(ascending=False))
ships_type_2019 = port2019_no_productcode['Scheepstype'].value_counts()
ships_class_2019 = port2019_no_productcode['Max lengte'].value_counts()
print("->Scheepstype for 2019 :")
print(ships_type_2019.sort_values(ascending=False))
print("->Max lengte for 2019 :")
print(ships_class_2019.sort_values(ascending=False))
#print("Check upper sum of ships matches passages 2015, OK," , ships_type_2015.sum())
#print("Check upper sum of ships matches passages 2019, OK," , ships_type_2019.sum())
#prct_ships_type_2015 = ships_type_2015['Scheepstype'].apply(lambda x : x / ships_type_2015.sum())
prct_ships_type_2015 = round((ships_type_2015 / ships_type_2015.sum())*100,4)
print("->Scheepstype for 2015 in %: ")
print(prct_ships_type_2015)
prct_ships_type_2019 = round((ships_type_2019 / ships_type_2019.sum())*100,4)
print("->Scheepstype for 2019 in %: ")
print(prct_ships_type_2019)
prct_ships_class_2015 = round((ships_class_2015 / ships_class_2015.sum())*100,4)
print("->Max lengte for 2015 in %: ")
print(prct_ships_class_2015)
prct_ships_class_2019 = round((ships_class_2019 / ships_class_2019.sum())*100,4)
print("->Max lengte for 2019 in %: ")
print(prct_ships_class_2019)


# TYPE OF SHIPS descending FOR STAY
print("\nType of ships, 'Max lengte' and 'Scheepstype', for Aankomst/Vertrek: ")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
port2015_no_productcode = port2015_no_productcode[port2015_no_productcode['Route categorie'] != 'Doorvaart']
port2019_no_productcode = port2019_no_productcode[port2019_no_productcode['Route categorie'] != 'Doorvaart']
ships_type_2015 = port2015_no_productcode['Scheepstype'].value_counts()
ships_class_2015 = port2015_no_productcode['Max lengte'].value_counts()
print("\nSTAY")
print("->Scheepstype for 2015: ")
print(ships_type_2015.sort_values(ascending=False))
print("->Max lengte for 2015 :")
print(ships_class_2015.sort_values(ascending=False))
ships_type_2019 = port2019_no_productcode['Scheepstype'].value_counts()
ships_class_2019 = port2019_no_productcode['Max lengte'].value_counts()
print("->Scheepstype for 2019 : ")
print(ships_type_2019.sort_values(ascending=False))
print("->Max lengte for 2019 :")
print(ships_class_2019.sort_values(ascending=False))
#print("Check upper sum of ships matches passages 2015, OK," , ships_type_2015.sum())
#print("Check upper sum of ships matches passages 2019, OK," , ships_type_2019.sum())
prct_ships_type_2015 = round((ships_type_2015 / ships_type_2015.sum())*100,4)
print("->Scheepstype for 2015 in %: ")
print(prct_ships_type_2015)
prct_ships_type_2019 = round((ships_type_2019 / ships_type_2019.sum())*100,4)
print("->Scheepstype for 2019 in %: ")
print(prct_ships_type_2019)
prct_ships_class_2015 = round((ships_class_2015 / ships_class_2015.sum())*100,4)
print("->Max lengte for 2015 in %: ")
print(prct_ships_class_2015)
prct_ships_class_2019 = round((ships_class_2019 / ships_class_2019.sum())*100,4)
print("->Max lengte for 2019 in %: ")
print(prct_ships_class_2019)



# PRODUCTS desscending
# DOORVAART AND STAY
print("\nProducts for Doorvaart + Aankomst/Vertrek: ")
products_2015 = port2015['Productcode'].value_counts()
prct_products_2015 = (products_2015 / products_2015.sum())*100
print("-> Productcode 2015 : 10 highest: ")
print(products_2015.head(10))
print("-> Productcode 2015 : 10 highest in % : ")
print(prct_products_2015.head(10))
products_2019 = port2019['Productcode'].value_counts()
prct_products_2019 = (products_2019 / products_2019.sum())*100
print("-> Productcode 2019 : 10 highest: ")
print(products_2019.head(10))
print("-> Productcode 2019 : 10 highest in % : ")
print(prct_products_2019.head(10))
# DOORVAART
print("\nProducts for Doorvaart: ")
products_2015 = port2015[port2015['Route categorie'] == 'Doorvaart']['Productcode'].value_counts()
prct_products_2015 = (products_2015 / products_2015.sum())*100
print("-> Productcode 2015 : 10 highest: ")
print(products_2015.head(10))
print("-> Productcode 2015 : 10 highest in % : ")
print(prct_products_2015.head(10))
products_2019 = port2019[port2019['Route categorie'] == 'Doorvaart']['Productcode'].value_counts()
prct_products_2019 = (products_2019 / products_2019.sum())*100
print("-> Productcode 2019 : 10 highest: ")
print(products_2019.head(10))
print("-> Productcode 2019 : 10 highest in % : ")
print(prct_products_2019.head(10))
# STAY
print("\nProducts for Aankomst/Vertrek: ")
products_2015 = port2015[port2015['Route categorie'] != 'Doorvaart']['Productcode'].value_counts()
prct_products_2015 = (products_2015 / products_2015.sum())*100
print("-> Productcode 2015 : 10 highest: ")
print(products_2015.head(10))
print("-> Productcode 2015 : 10 highest in % : ")
print(prct_products_2015.head(10))
products_2019 = port2019[port2019['Route categorie'] != 'Doorvaart']['Productcode'].value_counts()
prct_products_2019 = (products_2019 / products_2019.sum())*100
print("-> Productcode 2019 : 10 highest: ")
print(products_2019.head(10))
print("-> Productcode 2019 : 10 highest in % : ")
print(prct_products_2019.head(10))

#SCHEPEN VS PRODUCTEN


# ROUTE CATEGORIE
print("\nRoute categories : ")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
route_categorie_2015 = port2015_no_productcode['Route categorie'].value_counts()#['Route categorie'].value_counts()
prct_route_categorie_2015 = (route_categorie_2015 / route_categorie_2015.sum())*100
route_categorie_2019 = port2019_no_productcode['Route categorie'].value_counts()
prct_route_categorie_2019 = (route_categorie_2019 / route_categorie_2019.sum())*100
print("->Route categories for 2015:")
print(route_categorie_2015)
print("->Sum Route categories for 2015 :")
print(route_categorie_2015.sum())
print("->Route categories for 2015 in %:")
print(prct_route_categorie_2015)
print("->Route categories for 2019:")
print(route_categorie_2019)
print("->Sum Route categories for 2019:")
print(route_categorie_2019.sum())
print("->Route categories for 2019 in %:")
print(prct_route_categorie_2019)

# FOR ONE SHIP
"""
one = port2015[port2015['FD nummer ANO'] == 'FD_001']
print(one.head(15))
one = port2015[port2015['FD nummer ANO'] == 'FD_3456']
print(one.head(15))
"""

# RANK BOAT VISITS PER BOAT ID
#NO DOORVAART- ONLY AANKOMST/VERTREK
print("\nRank number of visits per boat : ")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
visits_per_id_2015 = port2015_no_productcode[port2015_no_productcode['Route categorie'] != 'Doorvaart']['FD nummer ANO'].value_counts()
print("Visits per boat for 2015, 10 highest")
print(visits_per_id_2015.head(10))
visits_per_id_2019 = port2019_no_productcode[port2019_no_productcode['Route categorie'] != 'Doorvaart']['FD nummer ANO'].value_counts()
print("Visits per boat for 2019, 10 highest")
print(visits_per_id_2019.head(10))
#TOTAL : DOORVAART + AANKOMST/VERTREK
print("\nRank number of visits per boat : ")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
visits_per_id_2015 = port2015_no_productcode['FD nummer ANO'].value_counts()
print("Visits per boat for 2015, 10 highest")
print(visits_per_id_2015.head(10))
visits_per_id_2019 = port2019_no_productcode['FD nummer ANO'].value_counts()
print("Visits per boat for 2019, 10 highest")
print(visits_per_id_2019.head(10))

# RANK MOST HERKOMST AND BESTEMMING HAVENS
print("\nRank Haven herkomst en bestemming:")
port2015_no_productcode = port2015.drop('Productcode',axis=1)
port2019_no_productcode = port2019.drop('Productcode',axis=1)
port2015_no_productcode = port2015_no_productcode.drop_duplicates(keep="first")
port2019_no_productcode = port2019_no_productcode.drop_duplicates(keep="first")
herkomst_2015 = port2015_no_productcode['Haven herkomst'].value_counts()
un_herkomst_2015 = len(port2015_no_productcode['Haven herkomst'].unique())
bestemming_2015 = port2015_no_productcode['Haven bestemming'].value_counts()
un_bestemming_2015 = len(port2015_no_productcode['Haven bestemming'].unique())
print("->Haven herkomst 2015, 10 most frequent:")
print(herkomst_2015.head(10))
print("->Haven bestemming 2015, 10 most frequent:")
print(bestemming_2015.head(10))
print("2015 : number aankomst ports :", un_herkomst_2015, " number bestemming ports :", un_bestemming_2015)
herkomst_2019 = port2019_no_productcode['Haven herkomst'].value_counts()
bestemming_2019 = port2019_no_productcode['Haven bestemming'].value_counts()
un_herkomst_2019 = len(port2019_no_productcode['Haven herkomst'].unique())
un_bestemming_2019 = len(port2019_no_productcode['Haven bestemming'].unique())
print("->Haven herkomst 2019, 10 most frequent:")
print(herkomst_2019.head(10))
print("->Haven bestemming 2019, 10 most frequent:")
print(bestemming_2019[0:10])#.head(10))
print("2019 : number aankomst ports :", un_herkomst_2019, " number bestemming ports :", un_bestemming_2019)


"""

voor je verrder rekent : alle shipmetrics behalve cargo moet je maken op unieke passage ship, NIET TWEE ONDER ELKAAR MET
GEWOON ANDERE CARGO!!!!!
verband tussen afmetingen, lengte en:/of dwt en type /passeren of aanmeren
verband tussen afmetingen en /of producten en aanmeren vs doorgaan
verband tussen afmetingen en passages
AFMETINGEN EN BESTEMMING?
AANTAL PRODUCTEN PER SHIP/TRIP
correlatie dwt en lengte
schip met meeste passages , meeste passages met effectieve cargo want niet alle FD counts zijn uniek, meerdere per trip
voor verschillende producten?
"""

