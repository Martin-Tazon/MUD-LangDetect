from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np
import pandas as pd



def compute_features(X_train, 
					 X_test, 
					 analyzer='char', 
					 max_features=None):
  '''
  Task: Compute a matrix of token counts given a corpus. 
		This matrix represents the frecuency any pair of tokens appears
		together in a sentence.
 
  Input: X_train -> Train sentences
		 X_test -> Test sentences
		 analyzer -> Granularity used to process the sentence 
					Values: {word, char}
		 tokenizer -> Callable function to apply to the sentences before compute.
  
  Output: unigramFeatures: Cout matrix
		  X_unigram_train_raw: Features computed for the Train sentences
		  X_unigram_test_raw: Features computed for the Test sentences 
  '''
  
  unigramVectorizer = CountVectorizer(analyzer=analyzer,
									  max_features=max_features,
									  ngram_range=(1,1))
  
  X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
  X_unigram_test_raw = unigramVectorizer.transform(X_test)
  unigramFeatures = unigramVectorizer.get_feature_names()
  return unigramFeatures, X_unigram_train_raw, X_unigram_test_raw
	


def compute_coverage(features, split, analyzer='char'):
	'''
	Task: Compute the proportion of a corpus that is represented by
	  	the vocabulary. All non covered tokens will be considered as unknown
	  	by the classifier.
	
	Input: features -> Count matrix
	  	 split -> Set of sentence 
	  	 analyzer -> Granularity level {'word', 'char'}
	
	Output: proportion of covered tokens
	'''
	total = 0.0
	found = 0.0
	for sent in split:
		#The following may be affected by your preprocess function. Modify accordingly
		sent = sent.split(' ') if analyzer == 'word' else list(sent)
		total += len(sent)
		for token in sent:
			if token in features:
				found += 1.0
	return found / total

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
	'''
	Task: Cast different types into numpy.ndarray
	Input: data ->  ArrayLike object
	Output: numpy.ndarray object
	'''
	data_type = type(data)
	if data_type == np.ndarray:
		return data
	elif data_type == list:
		return np.array(data_type)
	elif data_type == scipy.sparse.csr.csr_matrix:
		return data.toarray()
	print(data_type)
	return None	
  
def normalizeData(train, test):
	'''
	Task: Normalize data to train classifiers. This process prevents errors
		  due to features with different scale
	
	Input: train -> Train features
		   test -> Test features

	Output: train_result -> Normalized train features
			test_result -> Normalized test features
	'''
	train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
	test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
	return train_result, test_result


def plot_F_Scores(y_test, y_predict):
	'''
	Task: Compute the F1 score of a set of predictions given
		  its reference

	Input: y_test: Reference labels 
		   y_predict: Predicted labels

	Output: Print F1 score
	'''
	f1_micro = f1_score(y_test, y_predict, average='micro')
	f1_macro = f1_score(y_test, y_predict, average='macro')
	f1_weighted = f1_score(y_test, y_predict, average='weighted')
	print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))

def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
	'''
	Task: Given a set of reference and predicted labels plot its confussion matrix
	
	Input: y_test ->  Reference labels
		   y_predict -> Predicted labels
		   color -> [Optional] Color used for the plot
	
	Ouput: Confussion Matrix plot
	'''
	allLabels = list(set(list(y_test) + list(y_predict)))
	allLabels.sort()
	confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
	unqiueLabel = np.unique(allLabels)
	df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
	df_cm.index.name = 'Actual'
	df_cm.columns.name = 'Predicted'
	sn.set(font_scale=0.8) # for label size
	sn.set(rc={'figure.figsize':(15, 15)})
	sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size
	plt.show()


lang_mark = {'Romanian':(".","r"), 'Pushto':(",","g"), 'Russian':("o","b"), 'Urdu':("v","c"), 'Dutch':("^","m"), 'Latin':("<","y"), 'Spanish':(">","k"), 'Indonesian':(".","g"), 'Korean':(",","b"), 'French':("o","c"), 'English':("v","m"), 'Swedish':("^","y"), 'Tamil':("<","k"), 'Hindi':(">","r"), 'Estonian':(".","b"), 'Persian':(",","c"), 'Thai':("o","m"), 'Portugese':("v","y"), 'Chinese':("^","k"), 'Arabic':("<","r"), 'Japanese':(">","g"), 'Turkish':(".","c")}

def plotPCA(x_train, x_test,y_test, langs, markers=lang_mark, highlights=None, alpha_other=0.1):
	'''
	Task: Given train features train a PCA dimensionality reduction
		  (2 dimensions) and plot the test set according to its labels.
	
	Input: x_train -> Train features
		   x_test -> Test features
		   y_test -> Test labels
		   langs -> Set of language labels

	Output: Print the amount of variance explained by the 2 first principal components.
			Plot PCA results by language
			
	'''
	if highlights is None:
		highlights = langs
	pca = PCA(n_components=2)
	pca.fit(toNumpyArray(x_train))
	pca_test = pca.transform(toNumpyArray(x_test))
	print('Variance explained by PCA:', pca.explained_variance_ratio_)
	y_test_list = np.asarray(y_test.tolist())
	for lang in langs:
		if lang in highlights: 
			al = 1
		else:
			al = alpha_other
		pca_x = np.asarray([i[0] for i in pca_test])[y_test_list == lang]
		pca_y = np.asarray([i[1] for i in pca_test])[y_test_list == lang]
		if lang in markers:
			plt.scatter(pca_x,pca_y, label=lang, marker=markers[lang][0],color=markers[lang][1], alpha = al)
	plt.legend(loc="upper right")
	plt.show()

def word_char_uniqchar(raw):
	dfs = []
	dfdata = []
		
	# words
	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len))))
	#dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len).mean()))
	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len).std()))

	# chars
	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len))))
	#dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len).mean()))
	#dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len).std()))

	# Unique chars
	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len))))
	#dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len).mean()))
	#dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len).std()))

	#return pd.concat(dfs,axis=1).rename(columns={0:"word mean",1:"word std", 2:"char mean", 3:"char std", 4:"uniq char mean", 5:"uniq char std"}).round(2)
	return dfdata


def plot_unichar_vs_word(raw,markers=lang_mark, highlights=None, alpha_other=0.1):
	dd = word_char_uniqchar(raw)
	
	ddwords = raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len))
	dduchar = raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len))
	dw = ddwords.reset_index().rename(columns={"Text":"word"})
	du = dduchar.reset_index().rename(columns={"Text":"uni_char"}).drop(columns=["language"])
	df = pd.concat([dw,du],axis=1)
	
	#if highlights is None:
	#	highlights = lang_mark.keys
	#for lang in df["language"]:
	#	if lang in highlights: 
	#		al = 1
	#	else:
	#		al = alpha_other
	#	_x = df[df["language"]==lang]["uni_char"]
	#	_y = df[df["language"]==lang]["word"]
	#	plt.scatter(x="word", y="uni_char", label=lang, marker=markers[lang][0], color=markers[lang][1], alpha = al) 
	sns.scatterplot(data=df,x="uni_char",y="word",hue="language",linewidth=0)
	plt.show()

	return df

