import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
drug_dataset = pd.read_csv('./drug200.csv')

# Show head
drug_dataset.head()

sp = drug_dataset['Drug'].value_counts().plot(title='Drug instance count', figsize=(18, 6),
                                              kind='barh')

plt.savefig('./out/drug-distribution.pdf')