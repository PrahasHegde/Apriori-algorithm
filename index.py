#Association Algorithm -> Apriori Algo

"""Imagine yourself shopping in a supermarket, you are buying bread, and after putting the bread in your basket you move 
a little bit and you see butter just beside the bread counter, then you pick up the butter as well, then beside the butter 
counter you see mustard, now you will pick up the mustard and put it in your basket.If you just start noticing the placement of
products in the supermarket you will come to know that, they will place products like shampoo and conditioner together,
similarly, they will place bread and butter together it is because they have analyzed the purchasing patterns of 
customers to make people buy more product and increase their profit. This type of analysis is known as Market Basket Analysis"""

#imports
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Define the dataset by chatgpt
data = {
    'Transaction ID': [1, 2, 3, 4, 5],
    'Items': [['bread', 'milk'],
              ['bread', 'diaper', 'beer', 'eggs'],
              ['milk', 'diaper', 'beer', 'cola'],
              ['bread', 'milk', 'diaper', 'beer'],
              ['bread', 'milk', 'diaper', 'cola']]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)
print(df)

# Convert the list of items in each transaction into a string
df['Items'] = df['Items'].apply(lambda x: ','.join(x))
print(df)

# Apply one-hot encoding to convert the transaction data into a binary format
onehot = df['Items'].str.get_dummies(sep=',')
print(onehot)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(onehot, min_support=0.4, use_colnames=True)
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print(rules)


"""Antecedents — This column says what Item is being purchased first. See index 0, it says “beer” is being purchased first.

Consequents — This column says what Item is being purchased next after the first item(Antecedents), again see index 0,
 after purchasing “beer”, the “diaper” is being brought.

Antecedent support — This column gives the probability of how many times the antecedent is purchased, in index 0, 
 it says “beer” is being purchased 60% of the time.

Consequent support — Similarly, this will give the probability of how many times the consequent is purchased.

Support — This will give the probability of how many times both the Antecedent and Consequent are purchased together.
 See index 0, it says “beer” and “diaper” are purchased together 60% of the time.

Confidence — This column indicates the confidence of the association rule, which is the probability of finding the
 consequent item(s) in a transaction given that the antecedent item(s) are present. For example, the confidence of the 
 first rule is 1.00, indicating that ‘diaper’ is always purchased when ‘beer’ is purchased."""