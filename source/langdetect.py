import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import  preprocess

seed = 42
random.seed(seed)

#def word_char_uniqchar(raw):
#	dfs = []
#	dfdata = []
#	
#	# words
#	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len))))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len).mean()))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len).std()))
#
#	# chars
#	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len))))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len).mean()))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(list).apply(len).std()))
#
#	# Unique chars
#	dfdata.append(dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len))))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len).mean()))
#	dfs.append(raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len).std()))
#
#	#return pd.concat(dfs,axis=1).rename(columns={0:"word mean",1:"word std", 2:"char mean", 3:"char std", 4:"uniq char mean", 5:"uniq char std"}).round(2)
#	return dfdata
#
#def plot_unichar_vs_word(raw,markers=lang_mark, highlights=None, alpha_other=0.1):
#	dd = word_char_uniqchar(raw)
#	
#	ddwords = raw.groupby("language").apply(lambda x: x["Text"].str.split(" ").apply(len))
#	dduchar = raw.groupby("language").apply(lambda x: x["Text"].apply(set).apply(len))
#	dw = ddwords.reset_index().rename(columns={"Text":"word"})
#	du = dduchar.reset_index().rename(columns={"Text":"uni_char"}).drop(columns=["language"])
#	df = pd.concat([dw,du],axis=1)
#	
#	#df.plot.scatter(x="word",y="uni_char",color="language")
#	for lang in df["language"]:
#		_x = df[df["language"]==lang]["uni_char"]
#		_y = df[df["language"]==lang]["word"]
#		plt.scatter(x="word", y="uni_char", label=lang, marker=markers[lang][0], color=markers[lang][1], alpha = al)
#	#sns.scatterplot(data=df,x="uni_char",y="word",hue="language",linewidth=0)
#	plt.show()
#
#	return df

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", 
						help="Input data in csv format", type=str)
	parser.add_argument("-v", "--voc_size", 
						help="Vocabulary size", type=int)
	parser.add_argument("-a", "--analyzer",
						 help="Tokenization level: {word, char}", 
						type=str, choices=['word','char'])
	return parser


if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	raw = pd.read_csv(args.input)

	#raw=raw[(raw["language"] == "Chinese") | (raw["language"] == "Japanese")]
	
	# Languages
	languages = set(raw['language'])
	print('========')
	print('Languages', languages)
	print('========')

	# Split Train and Test sets
	X=raw['Text']
	y=raw['language']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
	
	print('========')
	print('Split sizes:')
	print('Train:', len(X_train))
	print('Test:', len(X_test))
	print('========')

	#### Get to know the data
	# Comment this to hide plot
	_ = plot_unichar_vs_word(raw,highlights=['Japanese', 'Spanish', 'Chinese', 'Arabic', 'English'])
	####

	# Preprocess text (Word granularity only)
	if args.analyzer == 'word':
		print(X_train)
		X_train, y_train = preprocess(X_train,y_train)
		print(X_train)
		X_test, y_test = preprocess(X_test,y_test)

	
	#raw_prep = pd.DataFrame({"language":y_train,"Text":X_train})
	#print(raw_prep)
	#df = word_char_uniqchar(raw_prep)
	
	#Compute text features
	features, X_train_raw, X_test_raw = compute_features(X_train, 
															X_test, 
															analyzer=args.analyzer, 
															max_features=args.voc_size)
	print(features)

	print('========')
	print('Number of tokens in the vocabulary:', len(features))
	print('Coverage: ', compute_coverage(features, X_test.values, analyzer=args.analyzer))
	print('========')


	#Apply Classifier  
	X_train, X_test = normalizeData(X_train_raw, X_test_raw)
	y_predict = applyNaiveBayes(X_train, y_train, X_test)

	print('========')
	print('Prediction Results:')	
	plot_F_Scores(y_test, y_predict)
	print('========')
	
	plot_Confusion_Matrix(y_test, y_predict, "Greens") 


	#Plot PCA
	print('========')
	print('PCA and Explained Variance:') 
	plotPCA(X_train, X_test,y_test, languages, highlights=['Japanese', 'Spanish', 'Chinese', 'Arabic', 'English']) 
	print('========')

