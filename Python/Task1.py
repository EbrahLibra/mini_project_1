## Step 1: Load the corpus using load files and make sure you set the encoding to latin1. (Task 1.3)

### Get and group the data
import sklearn.datasets
import pandas as pd


def load_files_of_bbc(category=None):
    """
    Gets the corpus of data
    :param category: The category/class of the the instance
    :return: The corpus, the length and list of file names
    """
    files_load = sklearn.datasets.load_files('../data/BBC', description="""
     D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.6
     """, categories=category, encoding='latin1')
    length = len(files_load.data)
    files_names = [fp[fp.find("\\") + 1:] for fp in files_load.filenames]
    return files_load, length, files_names


# all BBC data and size
allBBCFiles, allBBCDataSize, allBBC_filenames = load_files_of_bbc()

# #BBC business Data and size
businessFiles, businessDataSize, business_filenames = load_files_of_bbc('business')

# #BBC entertainment Data and size
entertainmentFiles, entertainmentDataSize, entertainment_filenames = load_files_of_bbc('entertainment')

# #BBC politics Data and size
politicsFiles, politicsDataSize, politics_filenames = load_files_of_bbc('politics')

# #BBC sport Data and size
sportFiles, sportDataSize, sport_filenames = load_files_of_bbc('sport')

# #BBC tech Data
techFiles, techDataSize, tech_filenames = load_files_of_bbc('tech')

## Step 2: Plot the distribution of the instances in each class and save the graphic in a file called BBC-distribution.pdf. (Task 1.2)

### Create Dataframe

row_name = 'Records Count'
allBBC_DF = pd.DataFrame({
    'Business': businessDataSize,
    'Entertainment': entertainmentDataSize,
    'Politics': politicsDataSize,
    'Sport': sportDataSize,
    'Tech': techDataSize
},
    index=[row_name]
)

print(allBBC_DF)

### Plot the distribution of the instances in each class

sp = allBBC_DF.loc[row_name].plot(by=allBBCFiles.target_names, title=row_name, figsize=(12, 6),
                               kind='barh')

### Save the graphic in a file called BBC-distribution.pdf

fig = sp.get_figure()
fig.savefig('../out/BBC-distribution.pdf')

